"""Vector store package: interface + FAISS implementation + factory.

    from latent_workspace.vector import build_vector_store
    store = build_vector_store(embedder)   # dim validated against settings
    store.add(unit_ids, embedder.embed(texts))
    hits  = store.search(query_vector, top_k=8)
"""

from __future__ import annotations

from .base import VectorHit, VectorStore, VectorStoreError
from .faiss_store import FaissVectorStore, ids_sidecar_path


def build_vector_store(embedder, settings=None) -> FaissVectorStore:
    """Create a store whose dimension matches the embedder AND settings.

    Catches the classic misconfiguration early: user switches
    ``DLRW_EMBEDDING_MODEL`` to a 768-dim model but leaves
    ``DLRW_EMBEDDING_DIM=384``.
    """
    from ..config import get_settings

    s = settings or get_settings()
    embedder_dim = int(getattr(embedder, "dim", 0) or 0)
    if embedder_dim <= 0:
        raise VectorStoreError(f"Embedder reports invalid dim: {embedder_dim!r}")
    if embedder_dim != int(s.embedding_dim):
        raise VectorStoreError(
            f"Embedder dim ({embedder_dim}) != DLRW_EMBEDDING_DIM "
            f"({s.embedding_dim}). Align them in your .env "
            f"(model={getattr(embedder, 'model_name', '?')})."
        )
    return FaissVectorStore(dim=embedder_dim)


__all__ = [
    "FaissVectorStore",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "build_vector_store",
    "ids_sidecar_path",
]
