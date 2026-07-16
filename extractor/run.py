"""CLI entry point for pass 1.

Walks the pdf-root for .pdf files (case-insensitive, recursive), optionally
filters by --match substrings (case-insensitive) so the Steppe/Baert
acceptance test can target named papers, processes up to --limit of them,
saves one row per PDF, prints each processed record when --show-records is
on, and finishes with a duplicate-DOI report, an OCR-skip list, an
abstract-fidelity-failed list, and a .md sibling report. Does not dedupe on
filename (WARP.md, "Deduplication").

One paper for review:

    .venv/bin/python -m extractor.run --pdf-root ~/phd/pdfs --limit 1

Acceptance test on two known papers:

    .venv/bin/python -m extractor.run --pdf-root ~/phd/pdfs --match Steppe Baert
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import db
from .extract import process_pdf

DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_PDF_ROOT = os.path.expanduser("~/phd/pdfs")
DEFAULT_DB = "data/review.db"


def _iter_pdfs(root):
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    return paths


def _match_filter(paths, terms):
    """Keep paths whose basename contains any term (case-insensitive)."""
    if not terms:
        return paths
    low = [t.lower() for t in terms]
    kept = []
    for path in paths:
        base = os.path.basename(path).lower()
        if any(t in base for t in low):
            kept.append(path)
    return kept


def _normalize_stem(name):
    """Whitespace- and punctuation-free lowercase stem, for .md sibling match."""
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _md_siblings(root, pdf_paths):
    """Return .md files whose stem matches a PDF stem, for manual deletion.

    Uses all discovered PDFs, not just the selected subset, so the report is
    complete even when --match or --limit narrowed the run. A PDF and its
    .md twin are the same paper in two formats (WARP.md, "Deduplication").
    """
    pdf_stems = {}
    for path in pdf_paths:
        base = os.path.basename(path)
        pdf_stems.setdefault(_normalize_stem(base), []).append(base)
    siblings = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".md"):
                stem = _normalize_stem(name)
                if stem in pdf_stems:
                    siblings.append(os.path.join(dirpath, name))
    siblings.sort()
    return siblings


def _print_record(record):
    """Print one record in full, including the raw abstract_verbatim."""
    print("=" * 78)
    print("FILENAME: " + str(record["filename"]))
    if record.get("needs_ocr"):
        print("NEEDS_OCR: true (no text layer; row skipped)")
    for key in [
        "doi", "title", "authors", "year", "journal", "volume", "pages",
        "paper_type", "crop_or_system", "perennial_or_annual_or_na",
        "setting", "location", "study_years", "research_question",
        "research_question_evidence", "stated_contribution",
        "stated_contribution_evidence", "stated_limitations",
        "data_sources", "validation_design", "validation_design_evidence",
        "performance_metrics", "topic_primary", "topics_secondary",
        "abstract_ocr_derived", "abstract_fidelity_failed",
        "validation_from_abstract",
        "provenance", "extraction_model",
        "extraction_date",
    ]:
        val = record.get(key)
        if val is None or val == "":
            val = "(null)"
        print(key + ": " + str(val))
    print("abstract_substituted: " + str(record.get("abstract_substituted")))
    print("ABSTRACT_VERBATIM (ASCII):")
    av = record.get("abstract_verbatim")
    print(av if av else "(null)")
    src = record.get("abstract_source")
    print("ABSTRACT_SOURCE (raw Unicode, first 400 chars):")
    print((src[:400] if src else "(null)") + (" ..." if src and len(src) > 400 else ""))
    print("=" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pass 1 extractor.")
    parser.add_argument("--pdf-root", default=DEFAULT_PDF_ROOT)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N PDFs (0 = all).",
    )
    parser.add_argument(
        "--match", nargs="+", default=None,
        help="Process only PDFs whose filename contains any of these "
        "substrings (case-insensitive). Use for the acceptance test, e.g. "
        "--match Steppe Baert.",
    )
    parser.add_argument(
        "--show-records", dest="show_records",
        action=argparse.BooleanOptionalAction, default=True,
        help="Print each processed record in full (--no-show-records to "
        "suppress for large runs).",
    )
    parser.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true",
        help="Skip PDFs that already have a row in the DB (by filename). "
        "Use to run the corpus in resumable chunks with --limit; only "
        "not-yet-processed PDFs are sent to the model.",
    )
    parser.add_argument(
        "--reset", dest="reset", action="store_true",
        help="Delete every row from review before processing. Use to "
        "re-run the whole corpus fresh after a schema or prompt change.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel worker threads for the I/O-bound "
        "OpenRouter calls (default 8). The calls are HTTP-bound, so "
        "threads are sufficient and the GIL is not a constraint. All "
        "SQLite writes stay on the main thread; workers only call "
        "process_pdf and return a record dict. If OpenRouter rate-limits "
        "at this concurrency, the 429 backoff handles it and the retry "
        "logging makes it visible so the count can be tuned down.",
    )
    parser.add_argument(
        "--provider-ignore", dest="provider_ignore", nargs="+", default=None,
        metavar="SLUG",
        help="OpenRouter provider slugs to exclude from routing for every "
        "request in this run (maps to provider.ignore in the request body). "
        "Use to route around a provider that consistently errors for a "
        "given model slug. Example: --provider-ignore amazon-bedrock",
    )
    parser.add_argument(
        "--provider-order", dest="provider_order", nargs="+", default=None,
        metavar="SLUG",
        help="OpenRouter provider slugs to try in priority order for every "
        "request in this run (maps to provider.order in the request body). "
        "Use to pin a specific provider. Example: --provider-order anthropic",
    )
    parser.add_argument(
        "--ocr-derived", dest="ocr_derived", action="store_true",
        help="Mark every row written in this run as abstract_ocr_derived=1. "
        "Use when the PDFs were pre-processed with ocrmypdf or similar before "
        "running pass 1, so the text layer (and therefore abstract_verbatim) "
        "comes from OCR rather than from the native PDF text. The fidelity "
        "check still runs, but both sides compare against the same OCR output "
        "and a pass is not evidence of accuracy against the printed page. "
        "Set this flag whenever you OCR papers and re-run; omit it for "
        "normal (born-digital) PDFs.",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY is not set. The extractor cannot "
            "run without it.",
            file=sys.stderr,
        )
        return 2

    root = os.path.abspath(os.path.expanduser(args.pdf_root))
    if not os.path.isdir(root):
        print("ERROR: pdf-root not found: " + root, file=sys.stderr)
        return 2

    all_paths = _iter_pdfs(root)
    total = len(all_paths)
    if total == 0:
        print("No PDFs found under " + root, file=sys.stderr)
        return 1

    matched = _match_filter(all_paths, args.match)
    if args.match and len(matched) == 0:
        print(
            "ERROR: --match " + " ".join(args.match)
            + " matched no PDFs in " + root,
            file=sys.stderr,
        )
        return 1

    limit = args.limit if args.limit and args.limit > 0 else len(matched)
    selected = matched[:limit]
    print(
        "Pass 1: " + str(len(selected)) + " of " + str(len(matched))
        + " matched (" + str(total) + " total) PDFs from " + root
    )
    print("Model: " + args.model)
    print("DB: " + args.db)
    print("Workers: " + str(args.workers))
    print("-" * 78, flush=True)

    conn = db.connect(args.db)
    if args.reset:
        conn.execute("DELETE FROM review")
        conn.commit()
        print("[reset] deleted all existing rows")
    processed = 0
    skipped = 0
    # Every process_pdf failure (TransliterationError on an unmapped
    # character, a JSON parse failure, or anything else) is collected here
    # in addition to the immediate stderr line, so it also appears in the
    # end-of-run summary. Without this, a handful of dropped papers can
    # scroll past unnoticed in a large --workers run with --show-records
    # on, and the paper silently has no row in the DB.
    failures = []
    # Resolve the skip list on the main thread before submitting to the
    # pool, so no sqlite3 connection crosses a thread boundary. The skip
    # check reads from the DB on this thread only.
    # known_dois is loaded once here and passed read-only to every worker.
    # Workers use it to skip the model call when the PDF's DOI hint matches
    # a DOI already in the DB (different filename, same paper).
    known_dois = db.get_dois(conn)
    to_process = []
    for path in selected:
        rel = os.path.relpath(os.path.abspath(path), root)
        if args.skip_existing and db.get_record(conn, rel) is not None:
            skipped += 1
            print("[skip] " + rel, flush=True)
        else:
            to_process.append(path)

    # Workers call process_pdf only (no DB access, no shared state). Each
    # returns a record dict (with _-prefixed instrumentation keys). The
    # main thread consumes futures as they complete and does every SQLite
    # write here, one row at a time, by a single thread.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_rel = {
            pool.submit(process_pdf, path, root, args.model,
                        provider_ignore=args.provider_ignore,
                        provider_order=args.provider_order,
                        known_dois=known_dois):
                os.path.relpath(os.path.abspath(path), root)
            for path in to_process
        }
        for future in as_completed(future_to_rel):
            rel = future_to_rel[future]
            try:
                record = future.result()
            except Exception as exc:
                stats = ""
                # Sourced from ChatError (failed call) or attached to a
                # JSON/transliteration exception after a successful call.
                s = getattr(exc, "_extraction_stats", None)
                if s:
                    fr = s.get("finish_reason")
                    nfr = s.get("native_finish_reason")
                    em = s.get("error_message")
                    pc = s.get("prompt_chars")
                    stats = (
                        " elapsed=%.1fs" % s["elapsed"]
                        + " total_elapsed=%.1fs" % s["total_elapsed"]
                        + " retries=" + str(s["retries"])
                        + " provider=" + str(s.get("provider") or "")
                        + " finish_reason=" + str(fr)
                        + " native_finish_reason=" + str(nfr)
                        + (" error_message=" + str(em) if em else "")
                        + (" prompt_chars=" + str(pc) if pc is not None else "")
                    )
                elif hasattr(exc, "attempts"):
                    # ChatError case: chat() failed after all retries.
                    attempts = exc.attempts
                    retries = len(attempts) - 1
                    last = attempts[-1] if attempts else {}
                    provider = last.get("provider")
                    fr = last.get("finish_reason")
                    nfr = last.get("native_finish_reason")
                    em = last.get("error_message")
                    stats = (
                        " retries=" + str(retries)
                        + " provider=" + str(provider or "")
                        + " finish_reason=" + str(fr)
                        + " native_finish_reason=" + str(nfr)
                        + (" error_message=" + str(em) if em else "")
                    )

                print("[ERROR] " + rel + ": " + str(exc) + stats, file=sys.stderr,
                      flush=True)
                failures.append((rel, str(exc) + stats))
                continue
            # DOI already in DB from a different filename — skip without writing.
            if record.get("_doi_skip"):
                skipped += 1
                doi = record.get("_doi_skip_doi", "?")
                print("[skip-doi] " + rel + " doi=" + doi, flush=True)
                continue
            if args.ocr_derived:
                record["abstract_ocr_derived"] = 1
            db.save_record(conn, record)
            processed += 1
            tags = []
            if record.get("needs_ocr"):
                tags.append("OCR-skip")
            if record.get("abstract_fidelity_failed") == 1:
                tags.append("FIDELITY-FAIL")
            if record.get("validation_from_abstract") == 1:
                tags.append("VA-FROM-ABSTRACT")
            tag = ",".join(tags) if tags else "ok"
            # Instrumentation line: filename, page count, prompt size,
            # elapsed seconds, retry count, provider. Explains a slow call
            # (retry + backoff, or a slow upstream provider) rather than
            # leaving it invisible.
            elapsed = record.get("_call_elapsed")
            total_elapsed = record.get("_call_total_elapsed")
            retries = record.get("_call_retries")
            provider = record.get("_call_provider")
            pages_n = record.get("_page_count")
            prompt_chars = record.get("_prompt_chars")
            # elapsed is the successful attempt only; total_elapsed spans
            # every attempt and backoff sleep. When they diverge, the gap
            # is time spent retrying, not time spent generating.
            stats = (
                " pages=" + str(pages_n) if pages_n is not None else ""
            ) + (
                " prompt_chars=" + str(prompt_chars)
                if prompt_chars is not None else ""
            ) + (
                " elapsed=%.1fs" % elapsed if elapsed is not None else ""
            ) + (
                " total_elapsed=%.1fs" % total_elapsed
                if total_elapsed is not None else ""
            ) + (
                " retries=" + str(retries) if retries is not None else ""
            ) + (
                " provider=" + str(provider) if provider else ""
            )
            print("[" + tag + "] " + rel + stats, flush=True)
            if args.show_records:
                _print_record(record)
                sys.stdout.flush()

    print("-" * 78)
    print("Processed: " + str(processed) + " / " + str(len(selected))
          + " (skipped " + str(skipped) + " existing, "
          + str(len(failures)) + " failed)")
    print("Total rows in DB: " + str(db.count_all(conn)))

    if failures:
        print("\nFailed (no row written; re-run with --skip-existing "
              "after fixing the cause):")
        for rel, err in failures:
            print("  - " + rel + ": " + err)

    dups = db.duplicate_report(conn)
    if dups:
        print("\nDuplicate DOIs (same paper, multiple files):")
        for doi, filenames in dups:
            print("  " + doi)
            for fn in filenames:
                print("    - " + fn)
    else:
        print("\nDuplicate DOIs: none among processed rows.")

    ocr = db.needs_ocr_list(conn)
    if ocr:
        print("\nNeeds OCR (no text layer, skipped):")
        for fn in ocr:
            print("  - " + fn)

    fid = db.fidelity_failed_list(conn)
    if fid:
        print("\nAbstract fidelity failed (word sequence not in page text):")
        for fn in fid:
            print("  - " + fn)

    vfa = db.validation_from_abstract_list(conn)
    if vfa:
        print("\nValidation design from abstract (methods section present):")
        for fn in vfa:
            print("  - " + fn)

    sibs = _md_siblings(root, all_paths)
    if sibs:
        print("\nMarkdown siblings (same paper as a PDF; delete if dup):")
        for s in sibs:
            print("  - " + os.path.relpath(s, root))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
