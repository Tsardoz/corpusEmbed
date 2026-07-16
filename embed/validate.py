"""Re-run the four asserts against a built embeddings.db. Exit non-zero on failure.

    python -m embed.validate [--db data/embeddings.db] [--export-dir export]

This is the standing guard: if any row ever drifts (wrong model tag, wrong dim,
title-only text, or a row/export count mismatch), this fails loud.
"""
import argparse
import os
import sqlite3
import sys

from . import asserts
from .specter2 import EMBED_DIM, EMBEDDING_MODEL_TAG

HERE = os.path.dirname(os.path.dirname(__file__))
OUT_DB = os.path.join(HERE, "data", "embeddings.db")
EXPORT_DIR = os.path.join(HERE, "export")


def _count_csv_rows(path):
    if not os.path.exists(path):
        return -1
    with open(path, encoding="ascii", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=OUT_DB)
    ap.add_argument("--export-dir", default=EXPORT_DIR)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM papers").fetchall()
    conn.close()

    failures = []
    for r in rows:
        try:
            asserts.assert_model_tag(r["embedding_model"])
            asserts.assert_dim(len(r["vector"]) // 4)          # float32 -> 4 bytes each
            asserts.assert_dim(int(r["embedding_dim"]))
            asserts.assert_abstract_included(
                bool((r["abstract"] or "").strip()), r["embedded_text"], r["abstract"])
        except AssertionError as e:
            failures.append(f'{r["key"]}: {e}')

    keys = [r["key"] for r in rows]
    n_export = _count_csv_rows(os.path.join(args.export_dir, "coverage.csv"))
    try:
        asserts.assert_one_pdf_one_row(len(rows), len(rows), n_export, keys)
    except AssertionError as e:
        failures.append(str(e))

    if failures:
        print("VALIDATE: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATE: OK  ({len(rows)} rows, all tagged {EMBEDDING_MODEL_TAG}, "
          f"all {EMBED_DIM}-dim, all abstract-included, coverage.csv rows={n_export})")


if __name__ == "__main__":
    main()
