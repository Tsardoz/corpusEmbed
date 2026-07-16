"""OpenRouter chat client over plain urllib (no SDK).

Sends a chat completion request and returns a ChatResult: the assistant
text plus wall-clock elapsed seconds, retry count, and the upstream
provider that served the request, for instrumentation. The model slug is
passed in by the caller and stored as extraction_model on every row. The
API key is read from the environment variable OPENROUTER_API_KEY; if it is
absent, fail loud (WARP.md, "Hard constraints" #2).

Two elapsed figures are reported, and the difference matters: `elapsed` is
the wall time of the single successful attempt; `total_elapsed` is the
wall time from the first attempt to the final result, including every
backoff sleep along the way. A call that hung, failed, backed off, and
then succeeded quickly reports a small `elapsed` and a large
`total_elapsed`; without the second figure that call is externally
indistinguishable from one that was simply fast, which defeats the point
of instrumenting this in the first place.
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT = 900
# Up from 3: at higher worker concurrency a sustained 429 burst can outlast
# a 3-attempt budget, silently dropping a paper from the DB.
_MAX_ATTEMPTS = 5
_MAX_BACKOFF = 60


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. The extractor cannot run "
            "without it (WARP.md, Hard constraints #2)."
        )
    return key



class ChatError(RuntimeError):
    """Raised when a chat call fails after all retries.

    Carries the last error message and the full attempts list.
    """

    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


class ChatResult:
    """Result of a chat call plus instrumentation stats.

    content: the assistant message text.
    elapsed: wall-clock seconds for the successful urlopen attempt only.
    total_elapsed: wall-clock seconds from the first attempt to the final
        result, including every backoff sleep. This is the figure that
        answers "why did this call take so long" -- elapsed alone hides
        time spent retrying.
    retries: number of retries before success (0 = first try succeeded).
    provider: upstream provider name from the OpenRouter response, or
        None. The model slug is not pinned to a provider, so logging this
        surfaces a slow or overloaded upstream as a cause of a slow call.
    finish_reason: OpenRouter-normalized finish reason from choices[0]
        ('stop', 'length', 'content_filter', 'tool_calls', 'error', or
        None). 'length' means the model hit its token cap mid-response.
    native_finish_reason: raw finish reason string from the provider,
        before OpenRouter normalization, or None.
    error_message: raw error detail string from the response body when
        finish_reason is 'error', or None. Sourced from choices[0].error
        or the top-level error field.
    attempts: list of per-attempt dicts (attempt number, status, elapsed,
        and error or provider).
    """
    __slots__ = (
        "content", "elapsed", "total_elapsed", "retries", "provider",
        "finish_reason", "native_finish_reason", "error_message", "attempts",
    )

    def __init__(
        self, content, elapsed, total_elapsed, retries, provider,
        finish_reason, native_finish_reason, error_message, attempts
    ):
        self.content = content
        self.elapsed = elapsed
        self.total_elapsed = total_elapsed
        self.retries = retries
        self.provider = provider
        self.finish_reason = finish_reason
        self.native_finish_reason = native_finish_reason
        self.error_message = error_message
        self.attempts = attempts


def chat(model, system, user, temperature=0.0, max_tokens=8000,
         provider_ignore=None, provider_order=None, response_format=None,
         thinking=None):
    """Return a ChatResult for a single chat turn.

    Retries up to _MAX_ATTEMPTS - 1 times on transient HTTP 429/5xx and
    network errors, with exponential backoff plus jitter, honoring a
    server-sent Retry-After header when present (chiefly on 429). Every
    retry is logged explicitly to stderr (attempt number, HTTP status or
    network reason, backoff slept) so a call that retried and then
    succeeded is not externally indistinguishable from one slow call.

    Two elapsed figures are tracked: `elapsed` is the successful attempt
    only; `total_elapsed` runs from the first attempt to the final result
    and includes every backoff sleep, so a call that spent most of its
    time retrying is visible as such rather than looking merely fast.

    provider_ignore: optional list of OpenRouter provider slugs to exclude
        from routing for this request (maps to provider.ignore in the body).
        Use to route around a provider that consistently errors.
    provider_order: optional list of OpenRouter provider slugs to try in
        priority order (maps to provider.order in the body). Use to pin a
        specific provider (e.g. ["anthropic"] routes exclusively through
        Anthropic's own API). Takes precedence over OpenRouter's default
        routing when set.
    response_format: optional dict sent verbatim as the response_format field
        in the request body (e.g. {"type": "json_object"} to enable JSON mode
        on models that support it). None means the field is omitted entirely,
        preserving the default text behaviour used in production.
    thinking: optional dict sent verbatim as the thinking field in the request
        body (e.g. {"type": "disabled"} for Kimi K2.6).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if thinking is not None:
        payload["thinking"] = thinking
    provider_overrides = {}
    if provider_ignore:
        provider_overrides["ignore"] = list(provider_ignore)
    if provider_order:
        provider_overrides["order"] = list(provider_order)
    if provider_overrides:
        payload["provider"] = provider_overrides
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + _api_key(),
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/literatureReview",
        "X-Title": "literatureReview pass1",
    }
    last_err = None
    retries = 0
    attempts = []
    run_start = time.monotonic()
    for attempt in range(_MAX_ATTEMPTS):
        req = urllib.request.Request(
            _ENDPOINT, data=body, headers=headers, method="POST"
        )
        attempt_start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                resp_headers = resp.headers
            elapsed = time.monotonic() - attempt_start
            total_elapsed = time.monotonic() - run_start
            data = json.loads(raw)
            provider = _provider(data, resp_headers)
            finish_reason, native_finish_reason = _finish_reason(data)
            error_message = _error_message(data)
            content = _extract_content(data)
            attempts.append({
                "attempt": attempt + 1,
                "status": "ok",
                "elapsed": elapsed,
                "provider": provider,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
                "error_message": error_message,
            })
            log_line = (
                "[openrouter] attempt %d ok provider=%s finish_reason=%s"
                " native=%s elapsed=%.1fs"
                % (attempt + 1, provider, finish_reason,
                   native_finish_reason, elapsed)
            )
            if error_message:
                log_line += " error_message=" + error_message
            print(log_line, file=sys.stderr, flush=True)
            return ChatResult(
                content=content,
                elapsed=elapsed,
                total_elapsed=total_elapsed,
                retries=retries,
                provider=provider,
                finish_reason=finish_reason,
                native_finish_reason=native_finish_reason,
                error_message=error_message,
                attempts=attempts,
            )
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - attempt_start
            status = exc.code
            last_err = _http_error(exc)
            retry_after = _retry_after_seconds(exc)
            attempts.append({
                "attempt": attempt + 1,
                "status": "http " + str(status),
                "elapsed": elapsed,
                "error": last_err,
            })
            if status in (429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
                backoff = _backoff_seconds(attempt, retry_after)
                retries += 1
                print(
                    "[openrouter] retry %d/%d after HTTP %d, backoff=%.1fs"
                    % (attempt + 2, _MAX_ATTEMPTS, status, backoff),
                    file=sys.stderr, flush=True,
                )
                _sleep(backoff)
                continue
            raise ChatError(last_err, attempts)
        except urllib.error.URLError as exc:
            elapsed = time.monotonic() - attempt_start
            reason = str(exc.reason)
            last_err = "OpenRouter network error: " + reason
            attempts.append({
                "attempt": attempt + 1,
                "status": "network",
                "elapsed": elapsed,
                "error": last_err,
            })
            if attempt < _MAX_ATTEMPTS - 1:
                backoff = _backoff_seconds(attempt, None)
                retries += 1
                print(
                    "[openrouter] retry %d/%d after network error: %s, "
                    "backoff=%.1fs"
                    % (attempt + 2, _MAX_ATTEMPTS, reason, backoff),
                    file=sys.stderr, flush=True,
                )
                _sleep(backoff)
                continue
            raise ChatError(last_err, attempts)
    raise ChatError(last_err or "OpenRouter request failed", attempts)


def _backoff_seconds(attempt, retry_after):
    """Return the backoff to sleep before the next attempt.

    Honors a server-reported Retry-After if present (typically on 429),
    otherwise exponential backoff capped at _MAX_BACKOFF, plus jitter so
    concurrent workers retrying together do not resynchronize on the same
    schedule and re-collide.
    """
    if retry_after is not None:
        base = max(0.0, retry_after)
    else:
        base = min(_MAX_BACKOFF, 2 ** attempt)
    jitter = random.uniform(0, base * 0.25 + 0.25)
    return base + jitter


def _retry_after_seconds(exc):
    """Parse a numeric Retry-After header from an HTTPError, or None.

    Only the delay-seconds form is handled (OpenRouter does not send the
    HTTP-date form in practice); anything unparsable is ignored and the
    caller falls back to exponential backoff.
    """
    try:
        value = exc.headers.get("Retry-After")
    except Exception:
        value = None
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _sleep(seconds):
    time.sleep(seconds)


def _extract_content(data):
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            "OpenRouter returned no content: "
            + json.dumps(data)[:500]
        )


def _provider(data, headers):
    """Return the upstream provider name from the OpenRouter response, or None.

    The model slug is not pinned to a provider, so a slow or overloaded
    upstream is the other leading cause of a slow call. OpenRouter may
    report the serving provider in the JSON body (a top-level 'provider'
    object with a 'name', or a per-choice 'provider') or in response
    headers. This captures whichever is present.
    """
    if isinstance(data, dict):
        name = _provider_name(data.get("provider"))
        if name:
            return name
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            name = _provider_name(choices[0].get("provider"))
            if name:
                return name
    try:
        for key, value in headers.items():
            if "provider" in key.lower() and value:
                return str(value)
    except Exception:
        pass
    return None


def _provider_name(prov):
    """Extract a provider name from a string or an object with a 'name'."""
    if isinstance(prov, dict):
        name = prov.get("name") or prov.get("model")
        if name:
            return str(name)
    elif isinstance(prov, str) and prov.strip():
        return prov.strip()
    return None


def _finish_reason(data):
    """Return (finish_reason, native_finish_reason) from the first choice.

    OpenRouter normalizes finish_reason to one of: 'stop', 'length',
    'content_filter', 'tool_calls', 'error', or None. The raw provider
    value is available as native_finish_reason. 'length' means the model
    hit its token cap mid-response and the output is truncated.
    """
    try:
        choice = (data.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        native_finish_reason = choice.get("native_finish_reason")
        return finish_reason, native_finish_reason
    except Exception:
        return None, None


def _error_message(data):
    """Extract a human-readable error message from the OpenRouter response.

    When finish_reason is 'error', the response body may carry error detail
    in several locations:
      - choices[0].error.message  (OpenRouter per-choice error object)
      - choices[0].error          (string form)
      - error.message             (top-level OpenRouter error)
      - error                     (top-level string)
    Returns the first non-empty string found, or None.
    """
    try:
        if not isinstance(data, dict):
            return None
        # Per-choice error object (most specific)
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            err = choices[0].get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("msg") or err.get("error")
                if msg:
                    return str(msg)[:500]
            elif isinstance(err, str) and err.strip():
                return err.strip()[:500]
        # Top-level error object
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or err.get("error")
            if msg:
                return str(msg)[:500]
        elif isinstance(err, str) and err.strip():
            return err.strip()[:500]
    except Exception:
        pass
    return None


def _http_error(exc):
    try:
        detail = exc.read().decode("utf-8", "replace")
    except Exception:
        detail = ""
    return "OpenRouter HTTP " + str(exc.code) + ": " + detail[:500]
