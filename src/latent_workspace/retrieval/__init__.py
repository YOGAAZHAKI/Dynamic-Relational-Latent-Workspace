"""Subgraph retrieval: query -> relevant, scored, budgeted context.

    from latent_workspace.retrieval import SubgraphRetriever

    retriever = SubgraphRetriever(graph)
    context   = retriever.retrieve("why did prices rise?")

The retriever *reads* the KnowledgeGraph / Embedder / vector store via
their public interfaces; it never mutates them and never re-implements
their responsibilities.
"""

from __future__ import annotations

from .base import RetrievalError, estimate_tokens
from .subgraph import RetrievalScore, SubgraphRetriever

__all__ = [
    "RetrievalError",
    "RetrievalScore",
    "SubgraphRetriever",
    "build_retriever",
    "estimate_tokens",
]


def build_retriever(graph, settings=None, **overrides) -> SubgraphRetriever:
    """Factory: a retriever over an existing workspace graph."""
    return SubgraphRetriever(graph, settings=settings, **overrides)
