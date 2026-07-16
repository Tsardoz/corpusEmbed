"""Per-PDF pass-1 orchestration.

Extracts page-bounded text with poppler, detects a missing text layer
(needs_ocr -> skip), calls the OpenRouter model with the page-numbered
text, parses the JSON response, stores the model's raw Unicode abstract as
abstract_source (never edited), runs the fidelity check against
abstract_source and the raw page text, derives abstract_verbatim by
applying the fixed transliteration table (extractor/ascii.py), and
assembles the full DB record including pipeline-set fields (provenance,
extraction_model, extraction_date, filename, topic_* null).

Transliteration is deterministic code, not a model behaviour, so the
fidelity check compares abstract_source against the page text exactly,
character for character (WARP.md, "Fidelity check, automatic").
"""

import datetime
import json
import os
import re

from . import pdf
from .ascii import transliterate
from .db import dump_json
from .fidelity import abstract_is_faithful
from .openrouter import chat
from .prompt import build_prompt

_PROVENANCE = "extracted_not_verified"


def process_pdf(path, root, model, provider_ignore=None, provider_order=None,
                known_dois=None):
    """Return a record dict ready for db.save_record.

    path: absolute path to the PDF.
    root: absolute path to the pdf-root (used to compute the relative
        filename used as the primary key).
    model: OpenRouter model slug, stored verbatim as extraction_model.
    provider_ignore: optional list of OpenRouter provider slugs to exclude
        from routing for this request (forwarded to chat() as-is).
    provider_order: optional list of OpenRouter provider slugs to try in
        priority order (forwarded to chat() as-is).
    known_dois: optional frozenset of DOIs already in the DB (loaded once on
        the main thread before the worker pool starts). When the PDF's DOI
        hint matches an entry in this set the model call is skipped entirely
        and the record is returned with _doi_skip=True. This avoids wasting
        an API call on a paper whose DOI was already extracted under a
        different filename in a previous run.
    """
    abspath = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    rel = os.path.relpath(abspath, root_abs)
    basename = os.path.basename(abspath)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )

    base = {
        "filename": rel,
        "pdf_path": abspath,
        "provenance": _PROVENANCE,
        "extraction_model": model,
        "extraction_date": now,
        "topic_primary": None,
        "topics_secondary": None,
        # Instrumentation keys, consumed by run.py for the per-paper log
        # line and stripped before db.save_record. Defaults cover the
        # OCR-skip and failed-PDF paths that return before the chat call.
        # _call_elapsed is the successful attempt only; _call_total_elapsed
        # spans every attempt and backoff sleep, so a call that retried and
        # then succeeded quickly is not logged as merely fast.
        "_call_elapsed": None,
        "_call_total_elapsed": None,
        "_call_retries": None,
        "_call_provider": None,
        "_prompt_chars": None,
        "_page_count": None,
        # Full page-marked text (pdftotext -raw with [PAGE n] markers) and the
        # page-numbering scheme, persisted in pass 1 so the separate full-text
        # step and pass 4 never re-run pdftotext. None on the OCR-skip,
        # failed-PDF, and doi-skip paths.
        "full_text_pagemarked": None,
        "page_numbering": None,
    }

    try:
        pages = pdf.extract_pages(abspath)
    except Exception as exc:
        # Corrupt or unreadable PDF: record an OCR-skipped row rather than
        # crashing the whole run (WARP.md, fail loud is for config; a bad
        # file is logged and skipped so the run continues).
        base.update(_empty_record())
        base["needs_ocr"] = 1
        base["title"] = "(extraction failed: " + str(exc)[:120] + ")"
        return base

    if not pdf.has_text_layer(pages):
        base.update(_empty_record())
        base["needs_ocr"] = 1
        base["abstract_fidelity_failed"] = None
        base["_page_count"] = len(pages)
        return base

    candidate_doi = pdf.find_doi(pages)
    if known_dois and candidate_doi and candidate_doi in known_dois:
        base["_doi_skip"] = True
        base["_doi_skip_doi"] = candidate_doi
        return base

    # Persist the full page-marked text now, from the pages pass 1 already
    # extracted, so the separate full-text step and pass 4 never re-run
    # pdftotext. Stored in -raw mode (natural reading order, which is what
    # pass 4 reads); page numbering is detected from the -layout pages the
    # model saw. A -raw failure falls back to the layout pages.
    try:
        raw_pages = pdf.extract_pages_raw(abspath)
    except Exception:
        raw_pages = None
    printed = pdf.detect_printed_numbers(pages)
    base["page_numbering"] = "printed" if printed else "pdf_index"
    base["full_text_pagemarked"] = pdf.build_full_text(
        raw_pages if raw_pages is not None else pages, printed
    )

    system, user = build_prompt(pages, basename, candidate_doi)
    prompt_chars = len(system) + len(user)
    result = chat(model, system, user, temperature=0.0, max_tokens=8000,
                  provider_ignore=provider_ignore, provider_order=provider_order)
    base["_call_elapsed"] = result.elapsed
    base["_call_total_elapsed"] = result.total_elapsed
    base["_call_retries"] = result.retries
    base["_call_provider"] = result.provider
    base["_prompt_chars"] = prompt_chars
    base["_page_count"] = len(pages)

    try:
        obj = _parse_json(result.content)
        base.update(_fields_from_model(obj))
    except Exception as exc:
        # Attach instrumentation to the exception so run.py can log it
        # even on a failing path (e.g. JSON parse failure).
        exc._extraction_stats = {
            "retries": result.retries,
            "provider": result.provider,
            "elapsed": result.elapsed,
            "total_elapsed": result.total_elapsed,
            "finish_reason": result.finish_reason,
            "native_finish_reason": result.native_finish_reason,
            "error_message": result.error_message,
            "prompt_chars": prompt_chars,
        }
        raise

    base["needs_ocr"] = 0
    base["abstract_ocr_derived"] = 0
    base["abstract_substituted"] = _as_int(
        obj.get("abstract_substituted"), default=0
    )
    abstract_pages = obj.get("abstract_pages")
    base["abstract_fidelity_failed"] = _fidelity_flag(
        base.get("abstract_source"), pages, abstract_pages, abspath
    )
    base["validation_from_abstract"] = _validation_from_abstract_flag(
        base.get("validation_design"), obj, pages
    )
    return base


def _empty_record():
    """Null values for every model-derived column (OCR/failed rows)."""
    return {
        "doi": None, "title": None, "authors": None, "year": None,
        "journal": None, "volume": None, "pages": None,
        "abstract_source": None, "abstract_verbatim": None,
        "abstract_substituted": 0, "abstract_ocr_derived": 0,
        "abstract_fidelity_failed": None,
        "validation_from_abstract": 0,
        "paper_type": None, "crop_or_system": None,
        "perennial_or_annual_or_na": None, "setting": None,
        "location": None, "study_years": None, "research_question": None,
        "research_question_evidence": "not_stated",
        "stated_contribution": None,
        "stated_contribution_evidence": "not_stated",
        "validation_design": None,
        "validation_design_evidence": "not_stated",
        "stated_limitations": None, "data_sources": None,
        "performance_metrics": None,
    }


def _fidelity_flag(abstract_source, pages, abstract_pages, path):
    """Return 1/0/None for abstract_fidelity_failed.

    None when there is no abstract to check (so the flag is nullable). 1
    when abstract_source's character sequence is not a contiguous substring
    of the page(s) it claims to come from, or when those pages are missing,
    out of range, or non-adjacent. 0 when the abstract checks out against
    its claimed source pages. The check is exact now that transliteration
    is out of the model: abstract_source keeps Greek, accents, the degree
    sign, and dashes as printed, and is compared against the raw pdftotext
    output with the same characters (WARP.md, "Fidelity check, automatic").

    The haystack is ONLY the page or pages the abstract was drawn from, at
    most two adjacent pages to allow an abstract spanning a page break (or
    the single page holding the first intro paragraph when
    abstract_substituted is true), never the whole document. Comparing
    against every page would make the check far weaker than intended.

    Two extraction modes are tried, both on the SAME selected page indices:
    first -layout (what the model saw), then -raw (stream order) as a
    fallback. For two-column PDFs -layout interleaves the columns and
    breaks the contiguous-substring match even on a faithful abstract; -raw
    keeps each column contiguous. A genuine paraphrase fails on both modes,
    so this tolerates only layout interleaving, never an altered character
    sequence. If -raw extraction itself errors, the -layout result stands.
    """
    if not abstract_source:
        return None
    selected = _select_abstract_pages(pages, abstract_pages)
    if selected is None:
        return 1
    if abstract_is_faithful(abstract_source, selected):
        return 0
    try:
        raw_pages = pdf.extract_pages_raw(path)
    except Exception:
        return 1
    raw_selected = _select_abstract_pages(raw_pages, abstract_pages)
    if raw_selected is None:
        return 1
    return 0 if abstract_is_faithful(abstract_source, raw_selected) else 1


def _select_abstract_pages(pages, abstract_pages):
    """Return the list of page strings the abstract came from, or None.

    None means the claimed provenance is unusable: no pages reported, more
    than two pages, two non-adjacent pages, or a page number out of range.
    In every one of those cases the fidelity check cannot run and the row
    is flagged as failed (a misreported source page is itself a fidelity
    problem). At most two adjacent pages are returned (WARP.md).
    """
    nums = sorted(set(_coerce_page_numbers(abstract_pages)))
    if not nums:
        return None
    if len(nums) > 2:
        return None
    if len(nums) == 2 and (nums[1] - nums[0]) != 1:
        return None
    selected = []
    for n in nums:
        if n < 1 or n > len(pages):
            return None
        selected.append(pages[n - 1])
    return selected


def _coerce_page_numbers(value):
    """Coerce a model-reported abstract_pages value into a list of ints.

    Accepts an int, a float, a list of ints/floats/numeric strings, or a
    string containing digits (e.g. "1", "1-2", "1 and 2"). Anything else
    yields an empty list, which the caller treats as missing provenance.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append(int(v))
            elif isinstance(v, str):
                out.extend(int(x) for x in re.findall(r"\d+", v))
        return out
    if isinstance(value, str):
        return [int(x) for x in re.findall(r"\d+", value)]
    return []


def _validation_from_abstract_flag(validation_design, obj, pages):
    """Return 1/0 for validation_from_abstract.

    1 only when validation_design has a value, its evidence records the
    source section as Abstract, AND the paper contains a Methods or
    Materials and Methods section. The field is not nulled; the row is
    flagged and listed in the report. This catches validation_design filled
    from the abstract when the methods say otherwise (WARP.md).
    """
    if not validation_design:
        return 0
    evidence = obj.get("validation_design_evidence")
    section = None
    if isinstance(evidence, dict):
        section = evidence.get("section")
    elif isinstance(evidence, str):
        section = evidence
    if not isinstance(section, str):
        return 0
    if section.strip().lower() != "abstract":
        return 0
    if not pdf.has_methods_section(pages):
        return 0
    return 1


def _fields_from_model(obj):
    """Pull fields from the parsed model JSON and ASCII-transliterate them.

    abstract_source is the model's raw Unicode return, stored unedited.
    abstract_verbatim is derived from abstract_source by the fixed
    transliteration table (extractor.ascii.transliterate), which raises on
    any non-ASCII character not in the table so the table grows by
    exception. All other text fields are likewise transliterated to ASCII
    for storage (the model no longer transliterates; WARP.md).
    """
    authors = obj.get("authors")
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    elif not isinstance(authors, list):
        authors = []
    authors = [transliterate(a) for a in authors if isinstance(a, str)]

    year = obj.get("year")
    if isinstance(year, str) and year.strip().isdigit():
        year = int(year.strip())
    elif isinstance(year, float):
        year = int(year)
    if not isinstance(year, int):
        year = None

    # abstract_source: raw model return, never edited (no strip, no
    # transliteration). Only coerce non-strings to None.
    abstract_source = obj.get("abstract_source")
    if not isinstance(abstract_source, str):
        abstract_source = None
    # abstract_verbatim: deterministic ASCII transliteration of
    # abstract_source. Raises TransliterationError on unknown characters.
    abstract_verbatim = transliterate(abstract_source)
    if abstract_verbatim is not None:
        abstract_verbatim = abstract_verbatim.strip() or None

    fields = {
        "doi": transliterate(_clean_str(obj.get("doi"))),
        "title": transliterate(_clean_str(obj.get("title"))),
        "authors": dump_json(authors),
        "year": year,
        "journal": transliterate(_clean_str(obj.get("journal"))),
        "volume": transliterate(_clean_str(obj.get("volume"))),
        "pages": transliterate(_clean_str(obj.get("pages"))),
        "abstract_source": abstract_source,
        "abstract_verbatim": abstract_verbatim,
        "paper_type": transliterate(_clean_str(obj.get("paper_type"))),
        "crop_or_system": transliterate(_clean_str(obj.get("crop_or_system"))),
        "perennial_or_annual_or_na": transliterate(
            _clean_str(obj.get("perennial_or_annual_or_na"))
        ),
        "setting": transliterate(_clean_str(obj.get("setting"))),
        "location": transliterate(_clean_str(obj.get("location"))),
        "study_years": transliterate(_clean_str(obj.get("study_years"))),
        "research_question": transliterate(
            _clean_str(obj.get("research_question"))
        ),
        "research_question_evidence": _norm_evidence(
            obj.get("research_question_evidence")
        ),
        "stated_contribution": transliterate(
            _clean_str(obj.get("stated_contribution"))
        ),
        "stated_contribution_evidence": _norm_evidence(
            obj.get("stated_contribution_evidence")
        ),
        "stated_limitations": transliterate(
            _clean_str(obj.get("stated_limitations"))
        ),
        "data_sources": transliterate(_clean_str(obj.get("data_sources"))),
        "validation_design": transliterate(
            _clean_str(obj.get("validation_design"))
        ),
        "validation_design_evidence": _norm_evidence(
            obj.get("validation_design_evidence")
        ),
        "performance_metrics": transliterate(
            _clean_str(obj.get("performance_metrics"))
        ),
    }
    if fields["doi"] is not None and not fields["doi"].startswith("10."):
        # A DOI that does not start with 10. is not a DOI; null it.
        fields["doi"] = None
    return fields


def _clean_str(value):
    """Return a trimmed string for a value, or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _as_int(value, default=0):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes"):
        return 1
    return default


def _norm_evidence(value):
    """Normalise an evidence field to a stored ASCII string.

    A dict {quote, page, section} is JSON-encoded with quote/section
    transliterated to ASCII. The string "not_stated" (or None / anything not
    a dict) is stored as "not_stated".
    """
    if isinstance(value, dict):
        clean = {}
        for key in ("quote", "page", "section"):
            if key in value:
                val = value[key]
                if isinstance(val, str):
                    clean[key] = transliterate(val)
                else:
                    clean[key] = val
        return json.dumps(clean, ensure_ascii=True)
    if isinstance(value, str) and value.strip() and value != "not_stated":
        # The model returned a bare quote string; wrap it without a page.
        return json.dumps(
            {"quote": transliterate(value), "page": None, "section": None},
            ensure_ascii=True,
        )
    return "not_stated"


def _parse_json(content):
    """Parse the model's JSON object, tolerating code fences and surrounding prose.

    Uses brace-counting to locate the FIRST balanced '{...}' object rather than
    taking text from the first '{' to the last '}'. This closes two failure
    classes:
      - Two concatenated JSON objects: the second object's closing brace was
        previously picked up by rfind, causing json.loads to see extra data.
      - Trailing fence markers or prose after the object: ignored because we
        stop at the exact closing brace of the first object.
    """
    if content is None:
        raise RuntimeError("Model returned an empty response.")
    text = content.strip()
    if text.startswith("```"):
        # Drop the opening fence line and a trailing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    start = text.find("{")
    if start == -1:
        raise RuntimeError(
            "No JSON object found in model response: " + content[:500]
        )
    # Walk forward counting braces; stop at the brace that closes the first
    # top-level object. String literals are tracked so braces inside strings
    # are not counted. Escaped characters (\x) inside strings are skipped.
    depth = 0
    in_string = False
    end = -1
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2  # skip escaped character
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        i += 1
    if end == -1:
        raise RuntimeError(
            "No JSON object found in model response: " + content[:500]
        )
    json_text = text[start:end + 1]
    # Sanitize: replace literal ASCII control characters (except tab 0x09,
    # newline 0x0A, carriage return 0x0D, which are valid JSON whitespace)
    # with their \uXXXX escape sequences. pdftotext sometimes emits raw
    # control characters in extracted text; the model faithfully copies them
    # into abstract_source, making the response unparseable by json.loads.
    # json.loads decodes \uXXXX back to the character, so abstract_source
    # receives the original character and fidelity normalization drops it
    # (control chars are not letters or digits).
    json_text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        lambda m: "\\u{:04x}".format(ord(m.group())),
        json_text,
    )
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Model response was not valid JSON: " + str(exc) + " | "
            + content[:500]
        )
