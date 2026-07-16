"""SPECTER2 embedding -- the single text-preparation and embedding code path.

Both papers and bucket centroids go through `make_text` + `Specter2Embedder.embed`,
so there is exactly one preparation of text across the whole corpus (the invariant
this project exists to enforce). No preferred/fallback branching anywhere.
"""
import os
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MODEL_NAME = "allenai/specter2_base"
ADAPTER_REF = "allenai/specter2"
ADAPTER_NAME = "proximity"

# Written into every row and checked by the asserts.
EMBEDDING_MODEL_TAG = "specter2"
EMBED_DIM = 768

# The model + adapter are already in the local HF cache; stay offline by default
# so a build never depends on the network. Override by exporting HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---- ASCII transliteration (plain ASCII throughout) -------------------------

_SMART = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2032": "'", "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
})
_DASH = str.maketrans({
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2010": "-", "\u00ad": "",
})


def to_ascii(text):
    """Lossy, robust transliteration to plain ASCII.

    abstract_verbatim is already ASCII (guaranteed by literatureReview), so this is
    a no-op there; titles may still carry ligatures/diacritics, which this folds.
    """
    if not text:
        return ""
    text = text.translate(_SMART).translate(_DASH)
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.encode("ascii", "ignore").decode("ascii").split())


def make_text(title, body, sep):
    """The ONE text assembly for papers and centroids: `title <sep> body`.

    Matches the SPECTER2 convention of `title + [SEP] + abstract`. Inputs are
    ASCII-folded so the WordPiece tokenizer never sees stray Unicode.
    """
    return sep.join([to_ascii(title).strip(), to_ascii(body).strip()])


class Specter2Embedder:
    """allenai/specter2_base + proximity adapter, CLS-pooled 768-dim embeddings."""

    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        self.model.load_adapter(ADAPTER_REF, source="hf", load_as=ADAPTER_NAME, set_active=True)
        # set_active=True alone leaves the adapter inactive for the forward pass;
        # this line is what actually activates it. Verified below.
        self.model.set_active_adapters(ADAPTER_NAME)
        self.model.to(device)
        self.model.eval()
        active = str(self.model.active_adapters)
        assert ADAPTER_NAME in active, f"SPECTER2 proximity adapter not active: {active!r}"

    @property
    def sep_token(self):
        return self.tokenizer.sep_token

    def embed(self, texts, batch_size=16):
        """Embed raw strings -> float32 ndarray (n, 768)."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inp = self.tokenizer(batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=512)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
            vecs.append(cls)
        arr = np.vstack(vecs)
        assert arr.shape[1] == EMBED_DIM, f"expected {EMBED_DIM}-dim, got {arr.shape[1]}"
        return arr
