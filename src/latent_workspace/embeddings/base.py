"""Embedder interface shared by mock and real implementations.

An **embedding** converts text into a vector of floats so that
"similar meaning" becomes "nearby points" in geometric space.

    embedder = build_embedder(settings)
    vectors  = embedder.embed(["first sentence", "second sentence"])
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


class EmbeddingError(Exception):
    """Any failure while turning text into vectors."""


@runtime_checkable
class Embedder(Protocol):
    """Interface every embedder satisfies."""

    model_name: str
    dim: int  # length of every produced vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...
