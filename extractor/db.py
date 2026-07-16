"""SQLite schema and CRUD for the review table (raw SQL, no ORM).

One row per PDF. The primary key is the PDF path relative to the pdf-root
(for a flat ~/phd/pdfs this equals the basename), so duplicate filenames
that are the same paper produce separate rows and can be flagged by DOI
afterwards (WARP.md, "Deduplication"). All columns nullable except
filename, provenance, extraction_model, and extraction_date.

Joined to nothing. Local SQLite only; no remote backend.
"""

import json
import sqlite3
from pathlib import Path
from typing import Union

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review (
    filename TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT,
    authors TEXT,
    year INTEGER,
    journal TEXT,
    volume TEXT,
    pages TEXT,
    abstract_source TEXT,
    abstract_verbatim TEXT,
    abstract_substituted INTEGER,
    needs_ocr INTEGER,
    abstract_ocr_derived INTEGER,
    abstract_fidelity_failed INTEGER,
    validation_from_abstract INTEGER,
    paper_type TEXT,
    crop_or_system TEXT,
    perennial_or_annual_or_na TEXT,
    setting TEXT,
    location TEXT,
    study_years TEXT,
    research_question TEXT,
    research_question_evidence TEXT,
    stated_contribution TEXT,
    stated_contribution_evidence TEXT,
    validation_design TEXT,
    validation_design_evidence TEXT,
    stated_limitations TEXT,
    data_sources TEXT,
    performance_metrics TEXT,
    topic_primary TEXT,
    topics_secondary TEXT,
    provenance TEXT NOT NULL,
    extraction_model TEXT NOT NULL,
    extraction_date TEXT NOT NULL,
    pdf_path TEXT,
    full_text_pagemarked TEXT,
    page_numbering TEXT,
    evidence_status TEXT
);
"""

# Insert column order, kept in one place to stay in sync with save_record.
_COLUMNS = [
    "filename", "doi", "title", "authors", "year", "journal", "volume",
    "pages", "abstract_source", "abstract_verbatim", "abstract_substituted", "needs_ocr",
    "abstract_ocr_derived", "abstract_fidelity_failed", "validation_from_abstract",
    "paper_type", "crop_or_system", "perennial_or_annual_or_na", "setting",
    "location", "study_years", "research_question",
    "research_question_evidence", "stated_contribution",
    "stated_contribution_evidence", "validation_design",
    "validation_design_evidence", "stated_limitations", "data_sources",
    "performance_metrics", "topic_primary", "topics_secondary",
    "provenance", "extraction_model", "extraction_date", "pdf_path",
    "full_text_pagemarked", "page_numbering",
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to existing DBs.

    CREATE TABLE IF NOT EXISTS does not add columns to a table that already
    exists, so a column added to SCHEMA_SQL must also be added here for old
    databases. abstract_source is added nullable; existing rows keep NULL
    and are NOT backfilled from abstract_verbatim, because the table
    transliteration is not invertible (WARP.md).
    """
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(review)"
    ).fetchall()}
    if "abstract_source" not in cols:
        conn.execute("ALTER TABLE review ADD COLUMN abstract_source TEXT")
        conn.commit()
    if "abstract_ocr_derived" not in cols:
        conn.execute("ALTER TABLE review ADD COLUMN abstract_ocr_derived INTEGER")
        conn.commit()
    if "full_text_pagemarked" not in cols:
        conn.execute("ALTER TABLE review ADD COLUMN full_text_pagemarked TEXT")
        conn.commit()
    if "page_numbering" not in cols:
        conn.execute("ALTER TABLE review ADD COLUMN page_numbering TEXT")
        conn.commit()
    if "evidence_status" not in cols:
        conn.execute("ALTER TABLE review ADD COLUMN evidence_status TEXT")
        conn.commit()


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open a connection, ensure the review table exists, and migrate it."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(SCHEMA_SQL)
    conn.commit()
    _migrate(conn)
    return conn


def save_record(conn: sqlite3.Connection, record: dict) -> None:
    """Insert or update one row by filename (the primary key)."""
    placeholders = ", ".join(["?"] * len(_COLUMNS))
    assignments = ", ".join(col + " = excluded." + col for col in _COLUMNS)
    sql = (
        "INSERT INTO review (" + ", ".join(_COLUMNS) + ") "
        "VALUES (" + placeholders + ") "
        "ON CONFLICT(filename) DO UPDATE SET " + assignments
    )
    conn.execute(sql, tuple(record[col] for col in _COLUMNS))
    conn.commit()


def get_record(conn: sqlite3.Connection, filename: str):
    row = conn.execute(
        "SELECT * FROM review WHERE filename = ?", (filename,)
    ).fetchone()
    return dict(row) if row else None


def count_all(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM review").fetchone()[0]


def duplicate_report(conn: sqlite3.Connection):
    """Return a list of (doi, [filenames]) for DOIs appearing more than once.

    This is the handoff for the user to delete the loser files (WARP.md,
    "Deduplication"). DOIs that are null or unique are not listed.
    """
    rows = conn.execute(
        "SELECT doi, filename FROM review "
        "WHERE doi IS NOT NULL AND doi != '' "
        "ORDER BY doi, filename"
    ).fetchall()
    by_doi = {}
    for row in rows:
        by_doi.setdefault(row["doi"], []).append(row["filename"])
    return [
        (doi, filenames)
        for doi, filenames in by_doi.items()
        if len(filenames) > 1
    ]


def needs_ocr_list(conn: sqlite3.Connection):
    """Return filenames flagged needs_ocr (no text layer, skipped)."""
    rows = conn.execute(
        "SELECT filename FROM review WHERE needs_ocr = 1 ORDER BY filename"
    ).fetchall()
    return [row["filename"] for row in rows]


def fidelity_failed_list(conn: sqlite3.Connection):
    """Return filenames flagged abstract_fidelity_failed.

    abstract_fidelity_failed is null when there was no abstract to check
    (needs_ocr rows, or rows with no abstract and no substituted one). Only
    rows where the flag is explicitly 1 are returned.
    """
    rows = conn.execute(
        "SELECT filename FROM review WHERE abstract_fidelity_failed = 1 "
        "ORDER BY filename"
    ).fetchall()
    return [row["filename"] for row in rows]


def validation_from_abstract_list(conn: sqlite3.Connection):
    """Return filenames flagged validation_from_abstract.

    These rows have a validation_design whose evidence section is Abstract
    while the paper also contains a Methods section -- the silent failure
    mode the downstream critical analysis depends on catching (WARP.md).
    The validation_design field is NOT nulled; the row is only flagged.
    """
    rows = conn.execute(
        "SELECT filename FROM review WHERE validation_from_abstract = 1 "
        "ORDER BY filename"
    ).fetchall()
    return [row["filename"] for row in rows]


def get_dois(conn: sqlite3.Connection) -> frozenset:
    """Return a frozenset of all non-null DOIs currently stored in the DB.

    Used by run.py to build the pre-check set passed to process_pdf so that
    papers whose DOI was already extracted in a previous run are skipped
    before the expensive model call, not just by filename (--skip-existing).
    The set is loaded once on the main thread before the worker pool starts;
    it is read-only and therefore safe to share across threads.
    """
    rows = conn.execute(
        "SELECT DISTINCT doi FROM review WHERE doi IS NOT NULL AND doi != ''"
    ).fetchall()
    return frozenset(row["doi"] for row in rows)


def save_fulltext(
    conn: sqlite3.Connection,
    filename: str,
    full_text_pagemarked: str,
    page_numbering: str,
) -> None:
    """Write full_text_pagemarked and page_numbering for one row.

    Uses a targeted UPDATE rather than save_record so that Pass 1 fields
    are never touched by the fulltext population step.
    """
    conn.execute(
        "UPDATE review SET full_text_pagemarked = ?, page_numbering = ?"
        " WHERE filename = ?",
        (full_text_pagemarked, page_numbering, filename),
    )
    conn.commit()


def dump_json(value):
    """JSON-encode a list/dict for a TEXT column, or return None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True)
