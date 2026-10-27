"""Tests for the embedding layer (Chunk 3)."""

from __future__ import annotations

import math

import pytest

from latent_workspace.config import Settings
from latent_workspace.embeddings import (
    EmbeddingError,
    MockEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_build_embedder_mock_mode_uses_configured_dim():
    s = Settings(_env_file=None, mock_mode=True, embedding_dim=64)
    emb = build_embedder(s)
    assert isinstance(emb, MockEmbedder)
    assert emb.dim == 64


def test_build_embedder_real_mode_constructs_st_embedder():
    pytest.importorskip("sentence_transformers")
    s = Settings(_env_file=None, mock_mode=False)
    emb = build_embedder(s)  # may download the model once
    try:
        assert isinstance(emb, SentenceTransformerEmbedder)
        assert emb.dim > 0
    except EmbeddingError as exc:
        pytest.skip(f"embedding model unavailable: {exc}")


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------

def test_mock_embedder_is_deterministic_across_instances():
    a = MockEmbedder(dim=64).embed(["the cat sat on the mat"])
    b = MockEmbedder(dim=64).embed(["the cat sat on the mat"])
    assert a == b  # identical vectors, even in a different instance


def test_mock_embedder_output_shape():
    emb = MockEmbedder(dim=32)
    out = emb.embed(["one", "two", "three"])
    assert len(out) == 3
    assert all(len(v) == 32 for v in out)
    assert all(isinstance(x, float) for v in out for x in v)


def test_mock_embedder_preserves_input_order():
    emb = MockEmbedder(dim=16)
    out = emb.embed(["alpha", "beta", "alpha"])
    assert out[0] == out[2] != out[1]


def test_mock_embedder_similar_texts_score_higher_cosine():
    emb = MockEmbedder(dim=384)
    base, near, far = emb.embed(
        [
            "the stock market rallied strongly today",
            "stock prices rose sharply in trading today",
            "quantum physics experiments use particle accelerators",
        ]
    )
    assert cosine(base, near) > cosine(base, far)


def test_mock_embedder_empty_text_still_yields_valid_vector():
    vec = MockEmbedder(dim=16).embed([""])[0]
    assert len(vec) == 16
    assert any(x != 0.0 for x in vec)  # non-zero so it can be normalized


def test_mock_embedder_rejects_tiny_dim():
    with pytest.raises(EmbeddingError):
        MockEmbedder(dim=4)


def test_mock_embedder_rejects_none_input():
    with pytest.raises(EmbeddingError):
        MockEmbedder().embed(None)  # type: ignore[arg-type]


def test_mock_embedder_ignores_punctuation_and_case():
    emb = MockEmbedder(dim=64)
    assert emb.embed(["Hello, World!"]) == emb.embed(["hello world"])


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder: class logic via an injected fake model
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402


class _FakeSTModel:
    """Mimics the tiny slice of the sentence-transformers API we use."""

    def __init__(self, dim=4, rows=None, error=None):
        self._dim = dim
        self._rows = rows  # force a specific row count (None = len(texts))
        self._error = error
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        if self._error:
            raise RuntimeError(self._error)
        n = len(texts) if self._rows is None else self._rows
        rows = []
        for i in range(n):
            rows.append([float(i + 1)] + [float(j) for j in range(self._dim - 1)])
        arr = np.array(rows, dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            raise AssertionError("store, not model, must normalize")
        return arr


def test_st_embedder_uses_injected_model_and_reads_dim():
    fake = _FakeSTModel(dim=4)
    emb = SentenceTransformerEmbedder("fake-model", model=fake)
    assert emb.dim == 4
    out = emb.embed(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 4
    assert fake.calls[0]["texts"] == ["a", "b"]
    assert fake.calls[0]["convert_to_numpy"] is True


def test_st_embedder_empty_input_returns_empty_list_without_calling_model():
    fake = _FakeSTModel()
    emb = SentenceTransformerEmbedder("fake", model=fake)
    assert emb.embed([]) == []
    assert fake.calls == []


def test_st_embedder_raises_on_wrong_vector_count():
    fake = _FakeSTModel(dim=4, rows=5)  # returns 5 vectors for 2 texts
    emb = SentenceTransformerEmbedder("fake", model=fake)
    with pytest.raises(EmbeddingError, match="vectors for 2 texts"):
        emb.embed(["a", "b"])


def test_st_embedder_raises_on_wrong_dimension():
    class _WrongDim:
        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **k):
            return np.zeros((len(texts), 7), dtype=np.float32)

    emb = SentenceTransformerEmbedder("fake", model=_WrongDim())
    with pytest.raises(EmbeddingError, match="does not match model dim"):
        emb.embed(["a"])


def test_st_embedder_wraps_model_errors():
    fake = _FakeSTModel(error="cuda out of memory")
    emb = SentenceTransformerEmbedder("fake", model=fake)
    with pytest.raises(EmbeddingError, match="cuda out of memory"):
        emb.embed(["a"])


def test_st_embedder_rejects_invalid_model_without_dim():
    class _NoDim:
        def get_sentence_embedding_dimension(self):
            return 0

    with pytest.raises(EmbeddingError, match="invalid dimension"):
        SentenceTransformerEmbedder("fake", model=_NoDim())


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder: the REAL model (skips if unavailable)
# ---------------------------------------------------------------------------

def _real_embedder() -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except EmbeddingError as exc:
        pytest.skip(f"real embedder unavailable: {exc}")


def test_real_st_embedder_orders_similar_meanings():
    emb = _real_embedder()
    try:
        cat, kitten, car = emb.embed(
            [
                "a cat is sitting on the mat",
                "a small feline rests on the rug",
                "a car drives down the highway",
            ]
        )
    except EmbeddingError as exc:
        pytest.skip(f"model encode failed: {exc}")

    assert len(cat) == emb.dim
    assert cosine(cat, kitten) > cosine(cat, car)


def test_real_st_embedder_dim_matches_default_settings():
    emb = _real_embedder()
    s = Settings(_env_file=None)
    assert emb.dim == s.embedding_dim  # default: 384
