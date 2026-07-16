"""Build one 768-dim SPECTER2 centroid per bucket from buckets.yaml.

The centroid text goes through the SAME make_text + embed path as papers:
a synthetic title of the bucket's first three keywords, then [SEP], then the
bucket description (the text the YAML says is "what gets embedded").
"""
import os

import yaml

from .specter2 import make_text

BUCKETS_YAML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "buckets.yaml")


def load_buckets(path=BUCKETS_YAML):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bucket_text(bucket, sep):
    """`first-3-keywords <sep> description`, via the shared paper code path."""
    keywords = bucket.get("keywords") or []
    title = ", ".join(keywords[:3])
    description = " ".join((bucket.get("description") or "").split())
    return make_text(title, description, sep)


def build_centroids(embedder, data):
    """Return (slugs, vectors[n_buckets, 768]) in the YAML's bucket order."""
    slugs = list(data["buckets"].keys())
    texts = [bucket_text(data["buckets"][s], embedder.sep_token) for s in slugs]
    vecs = embedder.embed(texts)
    return slugs, vecs
