"""Build the clean embedding + 6-bucket scoring DB.

Reads review.db (read-only) and buckets.yaml, embeds every paper through ONE code
path (title + [SEP] + abstract_verbatim), builds six centroids through that same
path, scores cosine against all six, applies the floor rule, and writes a fresh
data/embeddings.db plus plain-ASCII exports. The four asserts run at write time.

    python -m embed.build [--device cpu] [--review-db ...] [--buckets ...]
                          [--out-db ...] [--export-dir ...]
"""
import argparse
import csv
import json
import os
import sqlite3

import numpy as np

from . import asserts
from .centroids import BUCKETS_YAML, build_centroids, load_buckets
from .inputs import REVIEW_DB, load_papers
from .score import assign, cosine_scores, ranked
from .specter2 import (EMBED_DIM, EMBEDDING_MODEL_TAG, Specter2Embedder,
                       make_text, to_ascii)

HERE = os.path.dirname(os.path.dirname(__file__))
OUT_DB = os.path.join(HERE, "data", "embeddings.db")
EXPORT_DIR = os.path.join(HERE, "export")
SYS_ID = "system_identification"

DDL = """
DROP TABLE IF EXISTS papers;
DROP TABLE IF EXISTS buckets;
DROP TABLE IF EXISTS meta;
CREATE TABLE papers (
  key TEXT PRIMARY KEY, filename TEXT, doi TEXT, title TEXT, year INTEGER,
  journal TEXT, abstract TEXT,
  embedding_model TEXT NOT NULL, embedding_dim INTEGER NOT NULL,
  embedded_text_included_abstract INTEGER NOT NULL,
  embedded_text TEXT NOT NULL, vector BLOB NOT NULL,
  assigned_bucket TEXT NOT NULL, top_score REAL NOT NULL, scores_json TEXT NOT NULL
);
CREATE TABLE buckets (
  slug TEXT PRIMARY KEY, folder TEXT, keywords TEXT, description TEXT, centroid BLOB NOT NULL
);
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _write_exports(export_dir, papers, slugs, scores, pvecs, assigned, tops):
    os.makedirs(export_dir, exist_ok=True)
    cov = os.path.join(export_dir, "coverage.csv")
    header = (["key", "doi", "year", "assigned_bucket", "top_score"] + slugs
              + ["title", "journal", "has_abstract"])
    with open(cov, "w", newline="", encoding="ascii", errors="ignore") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, p in enumerate(papers):
            row = ([p.key, p.doi or "", p.year if p.year is not None else "",
                    assigned[i], round(tops[i], 6)]
                   + [round(float(scores[i][j]), 6) for j in range(len(slugs))]
                   + [to_ascii(p.title), to_ascii(p.journal), int(p.has_abstract)])
            w.writerow(row)
    np.save(os.path.join(export_dir, "vectors.npy"), pvecs.astype("float32"))
    with open(os.path.join(export_dir, "keys.txt"), "w", encoding="ascii", errors="ignore") as f:
        f.write("\n".join(p.key for p in papers) + "\n")
    return cov


def _count_csv_rows(path):
    with open(path, encoding="ascii", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _report(slugs, papers, scores, assigned, export_dir):
    lines = []
    order = slugs + ["unclassified"]
    counts = {s: 0 for s in order}
    for a in assigned:
        counts[a] = counts.get(a, 0) + 1
    lines.append("per-bucket counts (all six + unclassified):")
    for s in order:
        lines.append(f"  {s:<32s} {counts.get(s, 0)}")

    lines.append("")
    lines.append(f"{SYS_ID}: members (assigned) and near-members (2nd-highest):")
    hits = []
    for i, p in enumerate(papers):
        r = ranked(scores[i], slugs)
        is_member = assigned[i] == SYS_ID
        is_near = len(r) > 1 and r[1][0] == SYS_ID
        if is_member or is_near:
            tag = "MEMBER" if is_member else "near  "
            top2 = ", ".join(f"{slug}={sc:.4f}" for slug, sc in r[:2])
            hits.append((tag, to_ascii(p.title), top2))
    if hits:
        for tag, title, top2 in hits:
            lines.append(f"  [{tag}] {title[:90]}")
            lines.append(f"           {top2}")
    else:
        lines.append("  (none)")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(export_dir, "report.txt"), "w",
              encoding="ascii", errors="ignore") as f:
        f.write(text + "\n")


def _pdf_cosmetics(papers):
    """Item 5: report (do not fail on) missing / newline-broken pdf_path rows."""
    bad = []
    for p in papers:
        pp = p.pdf_path or ""
        if (not pp) or ("\n" in pp) or ("\r" in pp) or (not os.path.exists(pp)):
            bad.append(p)
    if bad:
        print(f"\ncosmetic (non-fatal, item 5): {len(bad)} rows have a missing or "
              f"newline-broken pdf_path; embeddings are unaffected:")
        for p in bad:
            reason = "newline in path" if ("\n" in (p.pdf_path or "") or "\r" in (p.pdf_path or "")) \
                else ("empty path" if not p.pdf_path else "file not found")
            print(f"  - {p.key}: {reason}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-db", default=REVIEW_DB)
    ap.add_argument("--buckets", default=BUCKETS_YAML)
    ap.add_argument("--out-db", default=OUT_DB)
    ap.add_argument("--export-dir", default=EXPORT_DIR)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    papers = load_papers(args.review_db)
    print(f"loaded {len(papers)} papers from {args.review_db} (read-only)")

    data = load_buckets(args.buckets)
    floor = float(data.get("floor", 0.5))
    unclassified = data.get("unclassified_folder", "unclassified")

    embedder = Specter2Embedder(device=args.device)
    sep = embedder.sep_token

    slugs, cvecs = build_centroids(embedder, data)
    print(f"built {len(slugs)} centroids: {', '.join(slugs)}")

    texts = [make_text(p.title, p.abstract, sep) for p in papers]
    prepared_abs = [to_ascii(p.abstract).strip() for p in papers]
    pvecs = embedder.embed(texts)
    print(f"embedded {pvecs.shape[0]} papers, dim {pvecs.shape[1]}")

    scores = cosine_scores(pvecs, cvecs)
    assigned, tops = [], []
    for i in range(len(papers)):
        a, t = assign(scores[i], slugs, floor, unclassified)
        assigned.append(a)
        tops.append(t)

    os.makedirs(os.path.dirname(args.out_db), exist_ok=True)
    conn = sqlite3.connect(args.out_db)
    conn.executescript(DDL)
    for i, s in enumerate(slugs):
        b = data["buckets"][s]
        conn.execute(
            "INSERT INTO buckets(slug,folder,keywords,description,centroid) VALUES(?,?,?,?,?)",
            (s, b.get("folder"), json.dumps(b.get("keywords") or []),
             to_ascii(" ".join((b.get("description") or "").split())),
             cvecs[i].astype("float32").tobytes()))
    for i, p in enumerate(papers):
        etext = texts[i]
        inc = bool(p.has_abstract and prepared_abs[i] and prepared_abs[i] in etext)
        asserts.assert_model_tag(EMBEDDING_MODEL_TAG)
        asserts.assert_dim(pvecs.shape[1])
        asserts.assert_abstract_included(p.has_abstract, etext, prepared_abs[i])
        scores_map = {slugs[j]: round(float(scores[i][j]), 6) for j in range(len(slugs))}
        conn.execute(
            "INSERT INTO papers(key,filename,doi,title,year,journal,abstract,"
            "embedding_model,embedding_dim,embedded_text_included_abstract,embedded_text,"
            "vector,assigned_bucket,top_score,scores_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.key, p.filename, p.doi, to_ascii(p.title), p.year, to_ascii(p.journal),
             prepared_abs[i], EMBEDDING_MODEL_TAG, EMBED_DIM, int(inc), etext,
             pvecs[i].astype("float32").tobytes(), assigned[i], tops[i],
             json.dumps(scores_map)))
    for k, v in [("embedding_model", EMBEDDING_MODEL_TAG), ("embedding_dim", str(EMBED_DIM)),
                 ("floor", str(floor)), ("n_papers", str(len(papers)))]:
        conn.execute("INSERT INTO meta(k,v) VALUES(?,?)", (k, v))
    conn.commit()

    cov = _write_exports(args.export_dir, papers, slugs, scores, pvecs, assigned, tops)

    n_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    asserts.assert_one_pdf_one_row(len(papers), n_rows, _count_csv_rows(cov),
                                   [p.key for p in papers])
    conn.close()

    _report(slugs, papers, scores, assigned, args.export_dir)
    _pdf_cosmetics(papers)
    print(f"\nwrote {args.out_db}")
    print(f"wrote {cov}, vectors.npy, keys.txt, report.txt in {args.export_dir}")
    print("asserts: all four passed at write time")


if __name__ == "__main__":
    main()
