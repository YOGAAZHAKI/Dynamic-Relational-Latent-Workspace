"""Deterministic, offline embedder for tests and demos.

How it works (bag-of-words hashing)
-----------------------------------
1. Lowercase the text and split it into word tokens.
2. For each token, SHA-1 hash it to a (bucket index, sign).
   SHA-1 (not the builtin ``hash``) because ``hash`` is randomized per
   process — the same text would give different vectors in a new run.
3. Add +1 / -1 into those buckets -> a sparse "signed word counts" vector.

Consequence: two texts sharing words produce vectors pointing in similar
directions, so cosine-based retrieval tests are *meaningful* — while
staying 100% free, offline and reproducible.

The returned vectors are NOT normalized; the vector store normalizes
before indexing so that inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from .base import EmbeddingError

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class MockEmbedder:
    """Hash-based embedder implementing the :class:`Embedder` protocol."""

    model_name = "mock-embedder"

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise EmbeddingError(f"Embedding dim must be >= 8, got {dim}")
        self.dim = int(dim)

    # -- interface -------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if texts is None:
            raise EmbeddingError("embed() requires a sequence of texts")
        return [self._embed_one(str(t)) for t in texts]

    # -- internals -------------------------------------------------------
    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:  # empty/whitespace text still needs a valid vector
            tokens = ["<empty>"]
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vector[bucket] += sign
        return vector
