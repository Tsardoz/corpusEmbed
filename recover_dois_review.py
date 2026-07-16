"""Recover DOIs for papers in review.db via CrossRef bibliographic query.

For each row in 'review' table where doi is missing:
  1. Query CrossRef works endpoint with title and first author.
  2. Score token-Jaccard similarity between stored title and candidate title.
  3. >= 0.90: auto-accept and UPDATE review.doi.
  4. 0.60-0.89: write to review CSV.
  5. Print summary counts.

Run from literatureReview/:
    CROSSREF_MAILTO=you@example.com python recover_dois_review.py [--dry-run]
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.request
import urllib.parse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = Path("data/review.db")
REVIEW_CSV = Path("export/doi_recovery_review.csv")
CROSSREF_WORKS = "https://api.crossref.org/works"
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "")
AUTO_ACCEPT  = 0.90
REVIEW_FLOOR = 0.60
RATE_SLEEP   = 2.0

# ── ASCII helpers ──────────────────────────────────────────────────────────────
_SMART = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u00ad": "",
})

def to_ascii(text):
    if not text: return ""
    text = text.translate(_SMART)
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    return stripped.encode("ascii", errors="replace").decode("ascii").replace("?", " ").strip()

def clean(text):
    return re.sub(r"\s+", " ", to_ascii(text or "")).strip()

def token_sim(a, b):
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def strip_jats(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

def crossref_query(title, first_author, rows=3, timeout=30, max_retries=4):
    query = title
    if first_author: query = title + " " + first_author
    params = {"query.bibliographic": query, "select": "DOI,title,author,abstract", "rows": rows}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    url = CROSSREF_WORKS + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "doi_recovery_review/1.0"}
    if CROSSREF_MAILTO:
        headers["User-Agent"] += f" (mailto:{CROSSREF_MAILTO})"

    last_exc = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data["message"]["items"]
            results = []
            for item in items:
                doi = item.get("DOI") or ""
                title_list = item.get("title") or []
                results.append({"doi": doi, "title": title_list[0] if title_list else "",
                                "abstract": strip_jats(item.get("abstract") or "")})
            return results
        except Exception as exc:
            last_exc = exc
            time.sleep(2 ** (attempt + 1))
    raise last_exc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    papers = [dict(r) for r in conn.execute(
        "SELECT filename, title, authors, abstract_verbatim FROM review WHERE doi IS NULL OR doi = ''"
    ).fetchall()]
    
    print(f"Found {len(papers)} papers missing DOIs.\n")
    review_rows = []
    n_accepted = 0
    n_review = 0
    n_unresolved = 0

    hdr = f"{'FILENAME':<40} {'SIM':>5} {'STATUS'}"
    print(hdr); print("-" * len(hdr))

    for p in papers:
        fn = p["filename"]
        title = p["title"] or ""
        authors = []
        try: authors = json.loads(p["authors"] or "[]")
        except: authors = [p["authors"]] if p["authors"] else []
        first_author = authors[0] if authors else ""

        try:
            candidates = crossref_query(title, first_author)
            time.sleep(RATE_SLEEP)
        except Exception as exc:
            print(f"{fn:<40} ERROR: {exc}")
            continue

        best_doi, best_score, best_title = "", 0.0, ""
        if candidates:
            for c in candidates:
                s = token_sim(title, c["title"])
                if s > best_score:
                    best_score, best_doi, best_title = s, c["doi"], c["title"]

        status = "UNRESOLVED"
        if best_score >= AUTO_ACCEPT:
            status = "ACCEPTED"
            if not args.dry_run:
                conn.execute("UPDATE review SET doi = ? WHERE filename = ?", (best_doi, fn))
                conn.commit()
            n_accepted += 1
        elif best_score >= REVIEW_FLOOR:
            status = "NEEDS_REVIEW"
            review_rows.append({"filename": fn, "stored_title": title, "candidate_doi": best_doi,
                                "candidate_title": best_title, "sim_score": round(best_score, 4)})
            n_review += 1
        else:
            n_unresolved += 1

        print(f"{fn[:39]:<40} {best_score:>5.3f} {status}")

    conn.close()
    if review_rows:
        REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(REVIEW_CSV, "w", newline="", encoding="ascii", errors="replace") as f:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
            w.writeheader(); w.writerows(review_rows)
        print(f"\nReview CSV: {REVIEW_CSV} ({len(review_rows)} rows)")
    print(f"\nAccepted: {n_accepted} | Review: {n_review} | Unresolved: {n_unresolved}")

if __name__ == "__main__":
    main()
