# corpusEmbed

Clean SPECTER2 embedding and six-bucket scoring of the ~100-paper review corpus.
Standalone project: it reads `~/phd/literatureReview/data/review.db` **read-only**
as the sole source of paper metadata, embeds each paper, scores it against six
topical bucket centroids, and writes a fresh `data/embeddings.db` plus ASCII
exports. It never writes to `literatureReview` and never touches `literatureSearch`.

## Why it exists
The previous embedding corpus drifted into three silent corruptions: a partial
ingest (57 of 106 papers), a mixed MiniLM/SPECTER2 embedding space, and title-only
vectors with the abstract left unused. This project rebuilds from scratch with one
consistent text preparation and four asserts that make any recurrence fail loud.

## Quick start
```
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m embed.build            # ingest -> embed -> score -> write + report
python -m embed.validate         # re-run the four asserts; non-zero on failure
```
SPECTER2 (`allenai/specter2_base` + proximity adapter) loads from the local HF
cache, so the build needs no network. CPU is the default; pass `--device cuda`
to use a GPU.

## Outputs
- `data/embeddings.db` — one row per paper: 768-dim vector, all six bucket scores,
  the assigned bucket, and the exact embedded text; plus a `buckets` table with the
  six centroids and a `meta` table.
- `export/coverage.csv` — one row per paper with all six scores (plain ASCII).
- `export/vectors.npy` + `export/keys.txt` — aligned vectors and keys.
- `export/report.txt` — per-bucket counts and the `system_identification`
  members / near-members.

## Buckets
`buckets.yaml` (this folder) is the authoritative source: six buckets, each with a
`description` that is embedded into a centroid. Adding a seventh is one more
centroid + re-score — no re-embedding of papers. See `WARP.md` for the invariants.
