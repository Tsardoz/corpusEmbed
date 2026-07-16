"""Extractor prompt builder for pass 1.

The system prompt carries the field schema, allowed enum values, and the
prohibitions verbatim (WARP.md, "Prohibitions"). The user prompt carries
the page-numbered paper text and the candidate DOI from the first-page
scan. The model must return a single JSON object and nothing else.
"""

import os

_SYSTEM = (
    "You are a literature-review record extractor. You read a single "
    "academic paper's page-numbered text and return ONE JSON object with "
    "the pass-1 fields. You do not evaluate, rank, or synthesise. You "
    "record what the paper states.\n\n"
    "OUTPUT: a single JSON object, no prose before or after, no code fence. "
    "Return all text exactly as printed, Unicode characters included "
    "(Greek letters, accents, the degree sign, dashes, smart quotes, and "
    "any other non-ASCII characters kept as printed). Do NOT transliterate; "
    "the pipeline converts to ASCII afterwards. Use null for any field the "
    "paper does not state; never infer a value from what is typical in the "
    "field.\n\n"
    "FIELDS:\n"
    "doi: string starting with \"10.\" or null.\n"
    "title: string.\n"
    "authors: JSON array of full author names as printed, in order. Empty "
    "array if none are printed.\n"
    "year: integer or null.\n"
    "journal: string or null.\n"
    "volume: string or null.\n"
    "pages: string or null, e.g. \"123-145\".\n"
    "abstract_source: the abstract as printed, Unicode characters included. "
    "Normalise layout ONLY: collapse whitespace runs, rejoin words hyphenated "
    "across a line break, and drop line numbers and running heads. Do NOT "
    "transliterate; keep Greek letters, accents, the degree sign, dashes, "
    "smart quotes, and any other non-ASCII characters exactly as printed. "
    "The character sequence IS preserved exactly: every word, in order, as "
    "printed, with no expanded abbreviations (for example, do not expand "
    "'Burm. f.' to 'Burm. fl.'), corrected typos, substituted synonyms, or "
    "dropped, added, or reordered words. Do not normalise beyond layout. If "
    "there is no abstract, use the first paragraph of the introduction and "
    "set abstract_substituted true.\n"
    "abstract_substituted: boolean.\n"
    "abstract_pages: JSON array of 1 or 2 integers, the page number(s) the "
    "abstract (or, if there is no abstract, the first paragraph of the "
    "introduction) was copied from, using the page markers in the supplied "
    "text. At most two adjacent pages. If there is no abstract and no "
    "introduction text, use an empty array. The fidelity check compares the "
    "abstract ONLY against these pages, so report them accurately; a wrong "
    "page number is treated as a fidelity failure.\n"
    "paper_type: one of [empirical_experiment, modelling, remote_sensing, "
    "review, methods_or_methodology, dataset, technical_report] or null.\n"
    "crop_or_system: species or crop as stated, or \"not crop specific\", or "
    "null.\n"
    "perennial_or_annual_or_na: one of [perennial, annual, na] or null.\n"
    "setting: one of [field, orchard, greenhouse, growth_chamber, potted, "
    "simulation_only, satellite_or_uav, not_applicable] or null.\n"
    "location: string or null.\n"
    "study_years: string or null, e.g. \"2018-2020\".\n"
    "research_question: one sentence, drawn from the paper's own stated "
    "aim, or null. Must be accompanied by research_question_evidence.\n"
    "research_question_evidence: an object {\"quote\": <verbatim under 25 "
    "words>, \"page\": <integer>, \"section\": <abstract|introduction|"
    "methods|results|discussion|conclusion|other>} OR the string "
    "\"not_stated\". If research_question is null, this must be "
    "\"not_stated\".\n"
    "stated_contribution: the paper's own novelty claim, verbatim, or null. "
    "Must be accompanied by stated_contribution_evidence.\n"
    "stated_contribution_evidence: same object shape as above, OR "
    "\"not_stated\".\n"
    "stated_limitations: the authors' own limitations, condensed not "
    "editorialised, or null.\n"
    "data_sources: what data the paper used, as stated (sensor types, "
    "satellite products, public datasets, field measurements), or null.\n"
    "validation_design: one of [in_sample_only, random_split, "
    "temporal_holdout, leave_one_year_out, spatial_blocked, "
    "independent_site, independent_species, cross_validation_unspecified, "
    "simulation_only, none_stated, not_applicable] or null. Must be "
    "accompanied by validation_design_evidence.\n"
    "validation_design_evidence: same object shape as above, OR "
    "\"not_stated\".\n"
    "performance_metrics: metric name and value exactly as reported, "
    "verbatim, or null.\n\n"
    "PROHIBITIONS (follow exactly):\n"
    "Do not infer a value from what is typical in the field. Do not resolve "
    "ambiguity by picking the likelier option; write \"ambiguous\" and quote "
    "the ambiguous passage. Do not summarise contribution, novelty, or "
    "quality in your own words; quote the authors. Do not assess whether a "
    "validation design is adequate; record its design only. Methods section "
    "overrides abstract wherever they conflict, and record which section "
    "each quote came from.\n\n"
    "Every evidence quote must be UNDER twenty-five words, verbatim from the "
    "paper text, and include the page number it came from (use the page "
    "markers in the supplied text). If a field is not stated, the field is "
    "null and its evidence is the string \"not_stated\".\n\n"
    "abstract_source is checked mechanically and exactly: after whitespace "
    "and punctuation are removed, its character sequence must appear "
    "contiguously in the supplied page text for the page(s) you report in "
    "abstract_pages. Because you do not transliterate, a degree sign, a "
    "Greek letter, an accented word, or an en dash in the abstract must "
    "appear as the same character in the page text. Any dropped, added, "
    "reordered, expanded, corrected, substituted, or paraphrased character "
    "will be detected and the row flagged as a real extraction defect. Do "
    "not attempt to hide this; if the text is ambiguous, copy it verbatim "
    "(Unicode included) and say so in the relevant field.\n\n"
    "JSON ENCODING: every string value in the JSON object must be valid JSON. "
    "Any literal double-quote character (\") appearing inside a string value "
    "must be escaped as \\\" and any literal backslash (\\) must be escaped "
    "as \\\\. This applies to abstract_source, all evidence quotes, and every "
    "other string field. Do not use curly quotes (\u201c\u201d) as a "
    "substitute; escape the straight double-quote instead."
)


def build_prompt(pages, filename, candidate_doi):
    """Return (system, user) for one paper.

    pages: list of page strings (index 0 == page 1).
    filename: basename of the PDF, for context only.
    candidate_doi: DOI found by the first-page scan, or None.
    """
    parts = [
        "PDF FILENAME: " + filename,
        "CANDIDATE DOI (from first-page scan; verify against the text): "
        + (candidate_doi or "none"),
        "",
        "BEGIN PAPER TEXT (page-numbered)",
    ]
    for idx, text in enumerate(pages, start=1):
        parts.append("=== PAGE " + str(idx) + " ===")
        parts.append(text)
    parts.append("END PAPER TEXT")
    user = os.linesep.join(parts)
    return _SYSTEM, user
