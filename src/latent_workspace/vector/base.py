"""Vector store interface.

A vector store answers: *"which stored vectors are nearest to this one?"*

Returns **(unit_id, cosine_score)** only — never relationship types.
Deciding SUPPORTS vs CONTRADICTS is the job of the relation classifier
(Chunk 4), not of similarity scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class VectorStoreError(Exception):
    """Any vector-store failure (dimensions, persistence, misuse)."""


@dataclass(frozen=True)
class VectorHit:
    """One search result: an InformationUnit id and its cosine score."""

    id: str
    score: float  # cosine similarity in [-1.0, 1.0]

    def as_tuple(self) -> tuple[str, float]:
        return (self.id, self.score)


@runtime_checkable
class VectorStore(Protocol):
    """Interface every vector store satisfies."""

    dim: int

    @property
    def size(self) -> int: ...

    def has(self, node_id: str) -> bool: ...

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> int:
        """Add vectors under stable InformationUnit ids.

        Returns how many were actually added (duplicates are skipped).
        """
        ...

    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[VectorHit]:
        """Nearest neighbors first, scores in [-1, 1] (cosine)."""
        ...

    def remove(self, node_id: str) -> bool: ...

    def clear(self) -> None: ...

    def save(self, path) -> None: ...
