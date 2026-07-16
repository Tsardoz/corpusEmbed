"""Automatic abstract fidelity check for pass 1.

The check compares abstract_source (the model's raw Unicode return, never
edited) against the raw pdftotext page text, character for character, with
NO transliteration on either side. Transliteration was moved out of the
model and into deterministic code (extractor/ascii.py) precisely so this
check can be exact: a degree sign, a Greek letter, an accented word, or an
en dash in the abstract must appear as the same character in the page text.
Any dropped, added, reordered, expanded, corrected, substituted, or
paraphrased token makes the normalized abstract stop being a contiguous
substring of the normalized page text, and the row is flagged.

normalize() reduces both sides by:
- NFC normalization (so precomposed and decomposed forms of the same glyph
  compare equal)
- casefold (Unicode-aware lowercase)
- keeping only Unicode letters (category L*) and numbers (category N*),
  dropping whitespace, punctuation, symbols, and marks

Keeping Greek letters and accented letters (category L) is deliberate: the
model is told to reproduce them as printed, so they must match the page
text exactly. Dropping symbols such as the degree sign on both sides is
also deliberate: it is a symbol, not a letter, so it carries no word
identity; stripping it on both sides compares the surrounding words. If the
model substituted a word for a symbol (e.g. "degrees" for the degree sign),
that word is a letter sequence present on one side and not the other, so
the check still flags it.

The haystack is ONLY the page(s) the abstract came from (selected by the
caller, at most two adjacent pages, or the single first-intro page when
abstract_substituted is true), never the whole document. The caller also
retries the match under pdftotext -raw when -layout interleaves two
columns; see extract._fidelity_flag.
"""

import unicodedata


def normalize(text):
    """Reduce text to a normalized key: NFC, casefold, keep letters and digits.

    Unicode letters (L*) and numbers (N*) are kept, including Greek and
    accented Latin. Everything else (whitespace, punctuation, symbols,
    marks) is dropped. None/empty returns "".
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] in ("L", "N")
    )


def abstract_is_faithful(abstract_source, pages):
    """True if abstract_source appears contiguously in the joined pages.

    abstract_source: the model's raw Unicode abstract (not transliterated).
    pages: the list of page strings to search (already narrowed to the
        abstract's source pages by the caller). They are concatenated.

    An empty/missing abstract returns True (nothing to check); the caller
    decides whether the flag should be null in that case.
    """
    needle = normalize(abstract_source)
    if not needle:
        return True
    haystack = normalize("\n".join(pages))
    return needle in haystack


def divergence(abstract_source, page_text):
    """Locate where abstract_source stops matching page_text.

    Returns None if the abstract is faithful. Otherwise returns a dict:
      matched_prefix_len: number of normalized characters that did match
      diverging_normalized: the normalized abstract suffix from the
        divergence point (up to 80 characters)
      mark: the normalized abstract with a "|" inserted at the divergence
        point (truncated around it for readability)

    Used by the diagnostic report so a human can read the diverging span
    (e.g. an expanded abbreviation such as "Burm. f." -> "Burm. fl.")
    rather than have it classified (WARP.md).
    """
    a = normalize(abstract_source)
    h = normalize(page_text)
    if not a or a in h:
        return None
    # Binary search the longest prefix of a that is a contiguous substring
    # of h. That prefix is the faithful part; the rest is the divergence.
    lo, hi, best = 0, len(a), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0 or a[:mid] in h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    diverging = a[best:best + 80]
    # Build a readable marker window around the divergence point.
    window_start = max(0, best - 40)
    window_end = min(len(a), best + 40)
    mark = a[window_start:best] + "|" + a[best:window_end]
    return {
        "matched_prefix_len": best,
        "diverging_normalized": diverging,
        "mark": mark,
    }
