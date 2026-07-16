"""Cosine scoring of papers against bucket centroids.

argmax over all six buckets; if the top cosine is below `floor` the paper is
routed to `unclassified` instead of being force-assigned. The full score vector
is retained -- the winning label alone is not the signal.
"""
import numpy as np


def l2norm(m):
    m = np.asarray(m, dtype="float32")
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def cosine_scores(paper_vecs, centroid_vecs):
    """Return (n_papers, n_buckets) cosine-similarity matrix."""
    return l2norm(paper_vecs) @ l2norm(centroid_vecs).T


def assign(scores_row, slugs, floor, unclassified="unclassified"):
    """Return (assigned_slug, top_score) applying the floor rule."""
    idx = int(np.argmax(scores_row))
    top = float(scores_row[idx])
    if top < floor:
        return unclassified, top
    return slugs[idx], top


def ranked(scores_row, slugs):
    """Return [(slug, score), ...] sorted by score descending."""
    order = np.argsort(scores_row)[::-1]
    return [(slugs[i], float(scores_row[i])) for i in order]
