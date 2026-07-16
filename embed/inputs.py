"""Load papers from this project's data/review.db, READ-ONLY.

review.db is the sole source of paper metadata. Opened with mode=ro so a build can
never mutate it. Dedup key is the DOI, falling back to nodoi-<sha256(filename)> for
rows with no DOI, so renamed files never re-ingest as duplicates.
"""
import hashlib
import os
import sqlite3

# Local copy of review.db lives in this project's data/ dir (copied from
# literatureReview, then extended in place by pass 1). Resolved relative to
# this file so it is independent of the current working directory, mirroring
# build.py's OUT_DB. corpusEmbed reads it read-only; pass 1 (extractor.run)
# is the only writer. Override with --review-db if needed.
REVIEW_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "review.db"
)


class Paper:
    __slots__ = ("key", "filename", "doi", "title", "abstract", "year",
                 "authors", "journal", "pdf_path", "has_abstract")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def synthetic_key(filename):
    h = hashlib.sha256((filename or "").encode("utf-8")).hexdigest()[:16]
    return f"nodoi-{h}"


def load_papers(db_path=REVIEW_DB):
    """Return a list[Paper]. Asserts dedup-key uniqueness (assert 4, part 1)."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"review.db not found at {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT filename, doi, title, authors, year, journal, "
            "abstract_verbatim, pdf_path FROM review"
        ).fetchall()
    finally:
        conn.close()

    papers, seen = [], {}
    for r in rows:
        doi = (r["doi"] or "").strip()
        key = doi if doi else synthetic_key(r["filename"])
        if key in seen:
            raise AssertionError(
                f"[assert 4] duplicate dedup key {key!r}: "
                f"{seen[key]!r} vs {r['filename']!r}")
        seen[key] = r["filename"]
        abstract = (r["abstract_verbatim"] or "").strip()
        papers.append(Paper(
            key=key,
            filename=r["filename"],
            doi=doi or None,
            title=(r["title"] or "").strip(),
            abstract=abstract,
            year=r["year"],
            authors=r["authors"],
            journal=(r["journal"] or "").strip(),
            pdf_path=r["pdf_path"],
            has_abstract=bool(abstract),
        ))
    return papers
