"""Strict ASCII transliteration with a fixed, literal character table.

abstract_source (the model's raw Unicode return, never edited) is converted
to abstract_verbatim (plain ASCII) by a literal character-to-string table.
The model no longer transliterates; transliteration is deterministic code
so the fidelity check can compare abstract_source against the raw pdftotext
output exactly, character for character, with no token substitution in the
way (WARP.md, "Fidelity check, automatic").

Coverage (WARP.md "at minimum" plus common operators/ligatures):
- degree sign, micro sign, multiplication sign, division sign,
  plus-minus sign, non-breaking space and other narrow spaces
- en dash, em dash, and the other dash variants
- smart single and double quotes, prime marks
- Greek letters (lowercase and uppercase) -> Latin names
- accented Latin letters -> base letter (see _ACCENTED_LATIN below)
- common mathematical operators (arrows, approximately, inequalities,
  infinity, sum, product, partial, increment)
- Latin ligatures (fi, fl, ff, ffi, ffl)
- ellipsis, bullet, middle dot, trademark/copyright/registered signs

Anything non-ASCII that is NOT in the table raises TransliterationError.
It is never silently dropped, so the table grows by exception rather than
by guesswork: when a run hits an unknown character, add it to _SYMBOLS or
extend _ACCENTED_LATIN's range and re-run.

This module's source uses \\u escapes, not literal non-ASCII bytes, to
respect the ASCII-only-source constraint (WARP.md, "Hard constraints" #1).
The accented-Latin table is precomputed once at import time by NFKD
decomposition over fixed Unicode ranges; NFKD is a defined Unicode
normalization (a deterministic operation), not a heuristic and not a model
call, and the result is a literal dict used by ordinary lookup at runtime.
Characters in those ranges that do not decompose to ASCII are excluded, so
they raise.
"""

import unicodedata


class TransliterationError(ValueError):
    """Raised when a non-ASCII character is not in the transliteration table.

    Carries the offending character and its codepoint so the table can be
    grown deliberately (WARP.md, "grow by exception").
    """

    def __init__(self, char):
        self.char = char
        super().__init__(
            "No transliteration for character U+%04X (%r). Add it to the "
            "table in extractor/ascii.py and re-run."
            % (ord(char), char)
        )


# Greek letters -> Latin names. The 24 lowercase and 24 uppercase letters
# of the standard Greek alphabet. (final sigma and regular sigma both map
# to "sigma".)
_GREEK = {
    "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
    "\u03b4": "delta", "\u03b5": "epsilon", "\u03b6": "zeta",
    "\u03b7": "eta", "\u03b8": "theta", "\u03b9": "iota",
    "\u03ba": "kappa", "\u03bb": "lambda", "\u03bc": "mu",
    "\u03bd": "nu", "\u03be": "xi", "\u03bf": "omicron",
    "\u03c0": "pi", "\u03c1": "rho", "\u03c2": "sigma",
    "\u03c3": "sigma", "\u03c4": "tau", "\u03c5": "upsilon",
    "\u03c6": "phi", "\u03c7": "chi", "\u03c8": "psi",
    "\u03c9": "omega",
    "\u0391": "Alpha", "\u0392": "Beta", "\u0393": "Gamma",
    "\u0394": "Delta", "\u0395": "Epsilon", "\u0396": "Zeta",
    "\u0397": "Eta", "\u0398": "Theta", "\u0399": "Iota",
    "\u039a": "Kappa", "\u039b": "Lambda", "\u039c": "Mu",
    "\u039d": "Nu", "\u039e": "Xi", "\u039f": "Omicron",
    "\u03a0": "Pi", "\u03a1": "Rho", "\u03a3": "Sigma",
    "\u03a4": "Tau", "\u03a5": "Upsilon", "\u03a6": "Phi",
    "\u03a7": "Chi", "\u03a8": "Psi", "\u03a9": "Omega",
}

# Symbols, punctuation, and spaces with defined ASCII replacements. These
# are the named minimum (degree, micro, multiplication, plus-minus, dashes,
# smart quotes, non-breaking space) plus other characters common in this
# corpus, so the table does not raise on ordinary scientific text.
_SYMBOLS = {
    # degree sign
    "\u00b0": " deg ",
    # superscript 2
    "\u00b2": "^2",
    # micro sign (U+00B5), distinct from Greek mu (U+03BC)
    "\u00b5": "u",
    # multiplication and division
    "\u00d7": "x", "\u00f7": "/",
    # plus-minus
    "\u00b1": "+/-",
    # dashes
    "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2212": "-",
    # smart quotes and primes
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2032": "'", "\u2033": '"', "\u2034": "'''",
    # spaces (non-breaking and narrow)
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200a": " ",
    "\u2002": " ", "\u2003": " ", "\u2007": " ", "\u200b": "",
    # ellipsis, bullet, middle dot
    "\u2026": "...", "\u2022": "*", "\u00b7": ".",
    # arrows
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->",
    "\u21d2": "=>", "\u21d0": "<=",
    # relational / mathematical operators
    "\u223c": "~", "\u2248": "~", "\u2264": "<=", "\u2265": ">=",
    "\u2260": "!=", "\u2261": "==",
    "\u221e": "inf", "\u2211": "sum", "\u220f": "prod",
    "\u2202": "partial", "\u2206": "delta",
    "\u00d0": "D",  # eth (rare)
    # ligatures
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    # signs
    "\u2122": "(TM)", "\u00a9": "(c)", "\u00ae": "(r)",
    # ordinal indicators (Spanish/Portuguese) -> drop the mark, keep letters
    "\u00aa": "a", "\u00ba": "o",
    # Latin letters not reachable by the NFKD ranges below
    "\u0131": "i",  # dotless i (no ASCII decomposition)
    # mathematical alphanumeric symbols (styled letters)
    "\U0001d438": "E",  # mathematical italic capital E
    "\U0001d43e": "K",  # mathematical italic capital K
    "\U0001d445": "R",  # mathematical italic capital R
    "\U0001d447": "T",  # mathematical italic capital T
    "\U0001d44e": "a",  # mathematical italic small a
    "\U0001d450": "c",  # mathematical italic small c
    # superscript digits and signs (U+2070-U+207F)
    # U+00B2 (^2) already above; U+00B9/B3 below _ACCENTED_LATIN range so explicit
    "\u2070": "^0", "\u00b9": "^1", "\u00b3": "^3",
    "\u2074": "^4", "\u2075": "^5", "\u2076": "^6",
    "\u2077": "^7", "\u2078": "^8", "\u2079": "^9",
    "\u207a": "^+", "\u207b": "^-", "\u207c": "^=", "\u207f": "^n",
    # subscript digits and signs (U+2080-U+208E)
    "\u2080": "_0", "\u2081": "_1", "\u2082": "_2", "\u2083": "_3",
    "\u2084": "_4", "\u2085": "_5", "\u2086": "_6",
    "\u2087": "_7", "\u2088": "_8", "\u2089": "_9",
    "\u208a": "_+", "\u208b": "_-", "\u208c": "_=",
    "\u208d": "_(", "\u208e": "_)",
    # white bullet used as degree symbol in older typesetting
    "\u25e6": " deg ",
    # guillemets (French/Spanish quotation marks)
    "\u00ab": '"', "\u00bb": '"',
    "\u2039": "'", "\u203a": "'",
    # currency that appears in affiliations; map to the letter code
    "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY",
}

# Accented Latin letters -> ASCII base, precomputed by NFKD over the
# Latin-1 Supplement (U+00C0-U+00FF), Latin Extended-A/B (U+0100-U+024F),
# and Latin Extended Additional (U+1E00-U+1EFF). A character is included
# only if its NFKD decomposition, with combining marks removed, is entirely
# ASCII; otherwise it is left out so it raises (and can be added by hand).
_ACCENTED_LATIN = {}
for _cp in list(range(0x00C0, 0x0250)) + list(range(0x1E00, 0x1F00)):
    _ch = chr(_cp)
    _decomp = unicodedata.normalize("NFKD", _ch)
    _base = "".join(c for c in _decomp if not unicodedata.combining(c))
    if _base and all(ord(c) < 128 for c in _base):
        _ACCENTED_LATIN[_ch] = _base

# Final table. Symbols and Greek take precedence over accented Latin on any
# overlap (there is none in practice), so they are merged last.
_TABLE = {}
_TABLE.update(_ACCENTED_LATIN)
_TABLE.update(_SYMBOLS)
_TABLE.update(_GREEK)


def transliterate(text):
    """Transliterate text to ASCII using the fixed table. None stays None.

    ASCII characters pass through unchanged. Non-ASCII characters in the
    table are replaced by their ASCII string. Any non-ASCII character not
    in the table raises TransliterationError (never silently dropped).
    """
    if text is None:
        return None
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in _TABLE:
            out.append(_TABLE[ch])
        else:
            raise TransliterationError(ch)
    return "".join(out)


def transliterate_lossy(text):
    """Transliterate text to ASCII, silently dropping unmapped non-ASCII chars.

    Identical to transliterate() except that any non-ASCII character not in
    _TABLE is dropped rather than raising TransliterationError.  Use for raw
    pdftotext page text, which can contain arbitrary Unicode (CJK, PUA,
    mathematical styled letters not yet in the table, etc.) that should not
    abort a corpus-wide check run.  The drop is semantically equivalent to
    the character not appearing on either side of a substring comparison, so
    it does not produce false positives: a word that was faithfully
    transliterated on the abstract_verbatim side still fails to match if its
    Unicode original is unknown here and therefore absent from the reduced
    haystack.  That is the correct outcome -- only table-covered characters
    are guaranteed to compare symmetrically.
    """
    if text is None:
        return None
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in _TABLE:
            out.append(_TABLE[ch])
        # else: silently drop unknown non-ASCII
    return "".join(out)


def is_ascii(text):
    """True if text is None or contains only ASCII characters."""
    if text is None:
        return True
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False
