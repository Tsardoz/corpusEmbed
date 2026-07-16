"""PDF text extraction via poppler subprocesses.

Uses pdftotext -layout with UTF-8 output and form-feed page separators so
page boundaries are preserved (every evidence quote carries a page number).
Uses pdfinfo for the page count. No Python PDF library; the poppler binaries
pdftotext and pdfinfo must be on PATH (see WARP.md, "Environment").
"""

import re
import shutil
import subprocess

_FORMFEED = "\f"
# A DOI is 10.<4-9 digits>/<suffix>. The suffix stops at whitespace and a few
# delimiters that are not part of a DOI.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)


def _require(binaries):
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        raise RuntimeError(
            "Missing required poppler binary(ies): " + ", ".join(missing)
            + ". Install poppler-utils (apt install poppler-utils)."
        )


def page_count(path):
    """Return the number of pages in the PDF via pdfinfo."""
    _require(["pdfinfo"])
    proc = subprocess.run(
        ["pdfinfo", "-enc", "UTF-8", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pdfinfo failed on " + str(path) + ": " + proc.stderr.strip()
        )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                break
    raise RuntimeError("pdfinfo did not report a page count for " + str(path))


def extract_pages(path):
    """Return a list of page strings; index 0 is page 1.

    pdftotext separates pages with a form-feed character. A single trailing
    empty page (common from pdftotext) is dropped.
    """
    _require(["pdftotext"])
    proc = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pdftotext failed on " + str(path) + ": " + proc.stderr.strip()
        )
    pages = proc.stdout.split(_FORMFEED)
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    return pages


def extract_pages_raw(path):
    """Return page strings in stream (reading) order via pdftotext -raw.

    Same form-feed page split and 1-based indexing as extract_pages, but
    without -layout. For two-column PDFs, -layout interleaves the columns
    line-by-line, which breaks the abstract fidelity contiguous-substring
    check; -raw reads each column contiguously, so the abstract block stays
    intact. Used only as the fidelity haystack fallback (see extract.py).
    Page count and indexing match extract_pages for the same PDF.
    """
    _require(["pdftotext"])
    proc = subprocess.run(
        ["pdftotext", "-raw", "-enc", "UTF-8", str(path), "-"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pdftotext -raw failed on " + str(path) + ": " + proc.stderr.strip()
        )
    pages = proc.stdout.split(_FORMFEED)
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    return pages


def has_text_layer(pages, min_chars=50):
    """Heuristic: a real text layer yields at least min_chars total."""
    return sum(len(p.strip()) for p in pages) >= min_chars


def find_doi(pages):
    """Scan the first one or two pages for a printed DOI.

    Returns the cleaned DOI (starting with "10.") or None. This is only a
    candidate handed to the extractor as a hint; the model verifies it
    against the full text.
    """
    head = "\n".join(pages[:2]) if pages else ""
    for match in _DOI_RE.finditer(head):
        doi = match.group(0).rstrip(".,;)]}>")
        doi = re.sub(
            r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE
        )
        if doi.startswith("10."):
            return doi
    return None


# Heading line that names a Methods / Materials and Methods / Methodology
# section, with an optional leading section number ("2 Methods", "2. Methods",
# "II. METHODS"). The line must be short and end after the heading word, so
# prose such as "the methods we used" does not match.
_METHODS_HEADING = re.compile(
    r"^(?:\d{1,2}[.)]?\s+|[ivxlcdm]+[.)]?\s+)?"
    r"(?:materials\s+and\s+methods|methods\s+and\s+materials|methodology"
    r"|methods?)\s*$",
    re.IGNORECASE,
)


def has_methods_section(pages):
    """True if any page has a line that looks like a Methods heading.

    Used by the validation_from_abstract flag to decide whether an
    abstract-sourced validation_design could have come from a methods
    section instead (WARP.md). Heuristic, not definitive; a false positive
    only over-flags a row for human review.
    """
    for page in pages:
        for line in page.splitlines():
            stripped = re.sub(r"\s+", " ", line).strip()
            if not stripped or len(stripped) > 50:
                continue
            if _METHODS_HEADING.match(stripped):
                return True
    return False


# Ported from the original fulltext.py so pass 1 can persist
# full_text_pagemarked in the same format a later pass 4 expects: pdftotext
# -raw text with [PAGE n] markers, page numbers printed when a reliable printed
# sequence is detected, otherwise the 1-based PDF index. Kept here (stdlib
# only) so pass 1 runs pdftotext once and downstream steps reuse the stored
# text instead of re-running it.
_BARE_INT = re.compile(r"^[-\u2013\u2014\s]*(\d{1,4})[-\u2013\u2014\s]*$")


def detect_printed_numbers(pages):
    """Return {1-based-pdf-index: printed-int} if reliable, else None.

    Looks at the first and last three non-empty lines of each page for a line
    whose entire content is a small integer (optionally wrapped in dashes or
    spaces). Reliable when at least 75% of pages yield a candidate AND at least
    80% of consecutive candidate pairs increment by exactly 1; otherwise None,
    and the caller falls back to the 1-based PDF index.
    """
    candidates = {}
    for i, page_text in enumerate(pages):
        pdf_idx = i + 1
        non_empty = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        probe = non_empty[:3] + non_empty[-3:]
        for line in probe:
            m = _BARE_INT.fullmatch(line)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 9999:
                    candidates[pdf_idx] = n
                    break
    if len(candidates) < max(1, len(pages) * 0.75):
        return None
    pairs = sorted(candidates.items())
    consec_total = sum(
        1 for (a, _), (b, _) in zip(pairs, pairs[1:]) if b == a + 1
    )
    consec_ok = sum(
        1 for (a, na), (b, nb) in zip(pairs, pairs[1:])
        if b == a + 1 and nb == na + 1
    )
    if consec_total == 0 or consec_ok / consec_total < 0.8:
        return None
    return candidates


def build_full_text(pages, printed_numbers):
    """Concatenate pages with [PAGE n] markers, each on its own line.

    n is the printed number when printed_numbers has an entry for that page,
    otherwise the 1-based PDF index. Mirrors the original fulltext.py builder
    so stored text stays compatible with what a ported pass 4 expects.
    """
    parts = []
    for i, page_text in enumerate(pages):
        pdf_idx = i + 1
        n = printed_numbers.get(pdf_idx, pdf_idx) if printed_numbers else pdf_idx
        parts.append("[PAGE " + str(n) + "]")
        parts.append(page_text)
    return "\n".join(parts)
