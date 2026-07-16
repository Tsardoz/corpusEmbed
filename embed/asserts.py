"""The four asserts -- the real fix.

These turn the four historical silent corruptions into loud failures. They are
enforced at write time (build) and re-run against the built DB (validate):

  1. embedding_model == 'specter2' on every row.
  2. every stored vector is exactly 768-dim.
  3. embedded text included the abstract whenever the paper has one.
  4. one PDF == one row == one export row, keyed on DOI (not path).
"""
from .specter2 import EMBED_DIM, EMBEDDING_MODEL_TAG


def assert_model_tag(tag):
    assert tag == EMBEDDING_MODEL_TAG, \
        f"[assert 1] embedding_model {tag!r} != {EMBEDDING_MODEL_TAG!r}"


def assert_dim(vec_len):
    assert vec_len == EMBED_DIM, f"[assert 2] vector dim {vec_len} != {EMBED_DIM}"


def assert_abstract_included(has_abstract, embedded_text, abstract):
    if has_abstract:
        assert abstract and abstract in embedded_text, \
            "[assert 3] paper has an abstract but the embedded text excluded it"


def assert_one_pdf_one_row(n_rows, n_embedded, n_export, keys):
    assert n_rows == n_embedded == n_export, (
        f"[assert 4] count mismatch: rows={n_rows} "
        f"embedded={n_embedded} export={n_export}")
    assert len(keys) == len(set(keys)), \
        "[assert 4] duplicate dedup keys (DOI/synthetic collision)"
