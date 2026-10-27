"""Shared helpers for subgraph retrieval (chunk 6).

Everything here is pure and deterministic — no graph access, no I/O:

- ``estimate_tokens`` / ``estimate_item_tokens``  budgeting heuristics
- ``cosine_similarity``                           query<->unit scoring
- ``validate_type_weights``                       config validation
"""

from __future__ import annotations

from ..models import RelationshipType

#: rough heuristic for English prose: ~4 characters per token
CHARS_PER_TOKEN = 4

#: fixed rendering overhead per context item (id, type, conf, newlines)
ITEM_OVERHEAD_TOKENS = 25
#: estimated cost of rendering one relation label line
RELATION_TOKENS = 12
#: estimated cost of rendering one provenance hop
PROVENANCE_TOKENS = 15


class RetrievalError(Exception):
    """Subgraph retrieval failure (bad query, embedder failure, ...)."""


def estimate_tokens(text: str) -> int:
    """Rough, deterministic token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_item_tokens(
    content: str,
    relation_count: int = 0,
    provenance_count: int = 0,
) -> int:
    """Estimated cost of one ``ContextItem`` in a rendered prompt."""
    return (
        estimate_tokens(content)
        + ITEM_OVERHEAD_TOKENS
        + RELATION_TOKENS * max(0, relation_count)
        + PROVENANCE_TOKENS * max(0, provenance_count)
    )


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for missing/zero vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def validate_type_weights(raw: dict[str, float]) -> dict[RelationshipType, float]:
    """Map string keys to ``RelationshipType``; reject unknown types.

    ``UNRELATED`` is rejected too: it is never stored as an edge, so a
    weight for it would be meaningless.
    """
    weights: dict[RelationshipType, float] = {}
    for key, value in raw.items():
        try:
            rtype = RelationshipType(key)
        except ValueError as exc:
            raise RetrievalError(
                f"Unknown relationship type in weights: {key!r}"
            ) from exc
        if rtype is RelationshipType.UNRELATED:
            raise RetrievalError(
                "UNRELATED can never be an edge, so it cannot carry a weight"
            )
        weights[rtype] = float(value)
    return weights
