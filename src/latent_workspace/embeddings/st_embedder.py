"""Sentence Transformers embedder (the real one).

Default model: ``sentence-transformers/all-MiniLM-L6-v2``
(384 dimensions, fast on CPU, ~90 MB, free, open source).

First use downloads the model from Hugging Face; afterwards it runs
entirely offline. A fake ``model`` can be injected in tests so the class
logic (batching, dimension checks) is verified without torch/network.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import EmbeddingError


class SentenceTransformerEmbedder:
    """Embeds text with a sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 32,
        model=None,  # injection point for tests (no torch/network needed)
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))

        if model is not None:
            self._model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is required for real embedding mode. "
                    "Install it:  python -m pip install sentence-transformers"
                ) from exc
            try:
                self._model = SentenceTransformer(model_name, device=device)
            except Exception as exc:  # network missing, bad model id, ...
                raise EmbeddingError(
                    f"Could not load embedding model '{model_name}' ({type(exc).__name__}): "
                    f"{exc}. Check your network (first run downloads the model) "
                    "or switch to DLRW_MOCK_MODE=true."
                ) from exc

        try:
            # sentence-transformers >= 6 renamed the method; support both
            getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dim = int(getter())
        except Exception as exc:
            raise EmbeddingError(f"Model does not expose its dimension: {exc}") from exc
        if self.dim <= 0:
            raise EmbeddingError(f"Model reported invalid dimension: {self.dim}")

    # -- interface -------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if texts is None:
            raise EmbeddingError("embed() requires a sequence of texts")
        if len(texts) == 0:
            return []

        try:
            raw = self._model.encode(
                [str(t) for t in texts],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,  # store normalizes at index time
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding call failed ({type(exc).__name__}): {exc}"
            ) from exc

        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim == 1:  # a single vector came back unsqueezed
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise EmbeddingError(f"Unexpected embedding shape {matrix.shape}")
        if matrix.shape[0] != len(texts):
            raise EmbeddingError(
                f"Model returned {matrix.shape[0]} vectors for {len(texts)} texts"
            )
        if matrix.shape[1] != self.dim:
            raise EmbeddingError(
                f"Vector dim {matrix.shape[1]} does not match model dim {self.dim}"
            )
        return matrix.astype(np.float32).tolist()
