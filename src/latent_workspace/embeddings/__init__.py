"""Embeddings package: interface + implementations + factory.

    from latent_workspace.embeddings import build_embedder
    embedder = build_embedder()                  # mock or real, per settings
    vectors  = embedder.embed(["text one", "text two"])
"""

from __future__ import annotations

from .base import Embedder, EmbeddingError
from .mock_embedder import MockEmbedder
from .st_embedder import SentenceTransformerEmbedder


def build_embedder(settings=None) -> Embedder:
    """Return the embedder for the current settings.

    ``DLRW_MOCK_MODE=true``  -> MockEmbedder        (offline, instant)
    ``DLRW_MOCK_MODE=false`` -> SentenceTransformerEmbedder (local model)
    """
    from ..config import get_settings

    s = settings or get_settings()
    if s.mock_mode:
        return MockEmbedder(dim=s.embedding_dim)
    return SentenceTransformerEmbedder(
        model_name=s.embedding_model,
        device=s.embedding_device,
        batch_size=s.embedding_batch_size,
    )


__all__ = [
    "Embedder",
    "EmbeddingError",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]
