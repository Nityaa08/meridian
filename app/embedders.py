"""
Embedders for the Extractor stage.

- "sentence-transformers" (default): all-MiniLM-L6-v2, a real semantic model.
  Needs PyTorch (requirements-ml.txt), about 5.6 GB installed.
- "hashing": a lexical stand-in in pure NumPy. Words and word pairs are hashed
  into a fixed-size vector with sub-linear term weights, so cosine similarity
  measures word overlap, not meaning. Used by the public demo, which runs on a
  serverless function with a 500 MB limit, and as the fallback when
  sentence-transformers is not installed.
"""
import logging
import re
import zlib

import numpy as np

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in into is it its of on or "
    "that the their this to was were what when where which who why will with".split()
)
_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Hashed bag of words and bigrams. encode() matches SentenceTransformer.encode()."""

    name = "hashing (lexical, no model)"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
        # Crude stemming so "attention" and "attentions" share a bucket.
        words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in self._tokens(text):
                h = zlib.crc32(tok.encode())
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 == 0 else -1.0  # signed hashing reduces collision bias
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, c in counts.items():
                out[i, idx] = np.sign(c) * (1.0 + np.log(abs(c))) if c else 0.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def load_embedder(backend: str, model_name: str, device: str | None):
    """Return an object with encode(texts). Falls back to hashing if PyTorch is missing."""
    if backend == "hashing":
        return HashingEmbedder()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed (pip install -r requirements-ml.txt); "
            "using the lexical hashing embedder instead."
        )
        return HashingEmbedder()
    model = SentenceTransformer(model_name, device=device)
    model.name = model_name
    return model
