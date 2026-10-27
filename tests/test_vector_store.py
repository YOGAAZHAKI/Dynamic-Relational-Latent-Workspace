"""Tests for the FAISS vector store (Chunk 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latent_workspace.vector import (
    FaissVectorStore,
    VectorHit,
    VectorStoreError,
    build_vector_store,
    ids_sidecar_path,
)


def unit(*values: float) -> list[float]:
    return list(values)


def unit_norm(*values: float) -> list[float]:
    """L2-normalize a tuple of floats (handy for expected-cosine math)."""
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


# ---------------------------------------------------------------------------
# construction & introspection
# ---------------------------------------------------------------------------

def test_store_starts_empty():
    store = FaissVectorStore(dim=3)
    assert store.size == 0
    assert store.ids == []
    assert store.has("iu_x") is False


def test_store_rejects_bad_dim():
    with pytest.raises(VectorStoreError):
        FaissVectorStore(dim=0)


# ---------------------------------------------------------------------------
# add / search basics
# ---------------------------------------------------------------------------

def test_search_returns_nearest_first_with_cosine_scores():
    store = FaissVectorStore(dim=3)
    store.add(["a", "b", "c"], [unit(1, 0, 0), unit(0, 1, 0), unit(-1, 0, 0)])

    hits = store.search(unit(1, 0, 0), top_k=3)
    assert [h.id for h in hits] == ["a", "b", "c"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)
    assert hits[2].score == pytest.approx(-1.0, abs=1e-5)
    assert all(isinstance(h, VectorHit) for h in hits)


def test_search_score_equals_manual_cosine():
    store = FaissVectorStore(dim=3)
    v1, v2 = unit(2, 3, 1), unit(1, 0, 2)
    store.add(["x"], [v1])
    hit = store.search(v2, top_k=1)[0]
    expected = sum(a * b for a, b in zip(unit_norm(*v1), unit_norm(*v2)))
    assert hit.score == pytest.approx(expected, abs=1e-6)


def test_non_normalized_vectors_are_normalized_by_the_store():
    """Cosine semantics must hold regardless of input magnitude."""
    store = FaissVectorStore(dim=2)
    store.add(["big", "orth"], [[100.0, 0.0], [0.0, 7.0]])
    hits = store.search([5.0, 0.0], top_k=2)
    assert hits[0].id == "big"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)


def test_query_vector_magnitude_does_not_change_score():
    store = FaissVectorStore(dim=3)
    store.add(["a"], [[0.6, 0.8, 0.0]])
    small = store.search([1.0, 0.0, 0.0], top_k=1)[0].score
    huge = store.search([1000.0, 0.0, 0.0], top_k=1)[0].score
    assert small == pytest.approx(huge, abs=1e-5)


def test_top_k_clamped_and_edge_cases():
    store = FaissVectorStore(dim=2)
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store.search([1.0, 1.0], top_k=99)) == 2   # clamped to size
    assert store.search([1.0, 1.0], top_k=0) == []         # nonsense request
    assert store.search([1.0, 1.0], top_k=-5) == []


def test_search_on_empty_store_is_safe():
    assert FaissVectorStore(dim=4).search([1.0] * 4, top_k=5) == []


def test_min_score_filters_weak_neighbors():
    store = FaissVectorStore(dim=3)
    store.add(["aligned", "orthogonal"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], top_k=2, min_score=0.5)
    assert [h.id for h in hits] == ["aligned"]


# ---------------------------------------------------------------------------
# duplicates & validation
# ---------------------------------------------------------------------------

def test_duplicate_id_is_skipped():
    store = FaissVectorStore(dim=2)
    added = store.add(["a"], [[1.0, 0.0]])
    assert added == 1
    added = store.add(["a"], [[0.0, 1.0]])  # re-add ignored, first wins
    assert added == 0
    assert store.size == 1
    assert store.search([1.0, 0.0], top_k=1)[0].score == pytest.approx(1.0, abs=1e-5)


def test_duplicate_inside_single_batch_keeps_first_occurrence():
    store = FaissVectorStore(dim=2)
    added = store.add(["a", "a", "b"], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert added == 2
    assert store.ids == ["a", "b"]  # first "a" (aligned with x-axis) kept


def test_id_vector_length_mismatch_raises():
    store = FaissVectorStore(dim=2)
    with pytest.raises(VectorStoreError, match="length mismatch"):
        store.add(["a", "b"], [[1.0, 0.0]])


def test_dimension_mismatch_on_add_raises():
    store = FaissVectorStore(dim=3)
    with pytest.raises(VectorStoreError, match="Dimension mismatch"):
        store.add(["a"], [[1.0, 0.0]])


def test_dimension_mismatch_on_search_raises():
    store = FaissVectorStore(dim=3)
    store.add(["a"], [[1.0, 0.0, 0.0]])
    with pytest.raises(VectorStoreError, match="Dimension mismatch"):
        store.search([1.0, 0.0], top_k=1)


def test_empty_id_is_rejected():
    store = FaissVectorStore(dim=2)
    assert store.add([""], [[1.0, 0.0]]) == 0


# ---------------------------------------------------------------------------
# remove / clear
# ---------------------------------------------------------------------------

def test_remove_known_id():
    store = FaissVectorStore(dim=2)
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert store.remove("a") is True
    assert store.size == 1 and not store.has("a")
    assert [h.id for h in store.search([0.0, 1.0], top_k=5)] == ["b"]


def test_remove_unknown_id_returns_false():
    assert FaissVectorStore(dim=2).remove("nope") is False


def test_remove_keeps_remaining_scores_correct():
    store = FaissVectorStore(dim=3)
    store.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    store.remove("b")
    assert "b" not in store.ids
    hits = store.search([0, 1, 0], top_k=3)
    assert {h.id for h in hits} == {"a", "c"}
    # neither remaining vector aligns with the y-axis query
    assert all(h.score == pytest.approx(0.0, abs=1e-5) for h in hits)


def test_clear_resets_everything():
    store = FaissVectorStore(dim=2)
    store.add(["a"], [[1.0, 0.0]])
    store.clear()
    assert store.size == 0
    assert store.search([1.0, 0.0]) == []
    store.add(["z"], [[1.0, 0.0]])  # still usable after clear
    assert store.size == 1


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    path = tmp_path / "vectors.faiss"
    store = FaissVectorStore(dim=3)
    store.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.save(path)

    assert path.exists() and ids_sidecar_path(path).exists()

    restored = FaissVectorStore.load(path)
    assert restored.dim == 3
    assert restored.ids == ["a", "b"]
    assert restored.size == 2

    before = store.search([1.0, 1.0, 0.0], top_k=2)
    after = restored.search([1.0, 1.0, 0.0], top_k=2)
    assert [h.as_tuple() for h in before] == [h.as_tuple() for h in after]

    # restored store is fully writable (id map rebuilt)
    restored.add(["c"], [[0.0, 0.0, 1.0]])
    assert restored.has("c")


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(VectorStoreError, match="not found"):
        FaissVectorStore.load(tmp_path / "missing.faiss")


def test_load_missing_sidecar_raises(tmp_path):
    import faiss

    path = tmp_path / "vectors.faiss"
    index = faiss.IndexFlatIP(3)
    index.add(np.zeros((1, 3), dtype="float32"))
    faiss.write_index(index, str(path))
    with pytest.raises(VectorStoreError, match="sidecar missing"):
        FaissVectorStore.load(path)


def test_load_sidecar_count_mismatch_raises(tmp_path):
    import json

    import faiss

    path = tmp_path / "vectors.faiss"
    index = faiss.IndexFlatIP(3)
    index.add(np.zeros((2, 3), dtype="float32"))
    faiss.write_index(index, str(path))
    ids_sidecar_path(path).write_text(
        json.dumps({"dim": 3, "ids": ["only-one"]}), encoding="utf-8"
    )
    with pytest.raises(VectorStoreError, match="Row mismatch"):
        FaissVectorStore.load(path)


def test_load_sidecar_dim_mismatch_raises(tmp_path):
    import json

    import faiss

    path = tmp_path / "vectors.faiss"
    index = faiss.IndexFlatIP(3)
    index.add(np.zeros((1, 3), dtype="float32"))
    faiss.write_index(index, str(path))
    ids_sidecar_path(path).write_text(
        json.dumps({"dim": 7, "ids": ["a"]}), encoding="utf-8"
    )
    with pytest.raises(VectorStoreError, match="Dim mismatch"):
        FaissVectorStore.load(path)


# ---------------------------------------------------------------------------
# factory dimension guard
# ---------------------------------------------------------------------------

def test_build_vector_store_accepts_matching_dim():
    from latent_workspace.embeddings import MockEmbedder

    store = build_vector_store(MockEmbedder(dim=48), _settings(dim=48))
    assert store.dim == 48


def test_build_vector_store_rejects_dim_mismatch():
    from latent_workspace.embeddings import MockEmbedder

    with pytest.raises(VectorStoreError, match="!= DLRW_EMBEDDING_DIM"):
        build_vector_store(MockEmbedder(dim=32), _settings(dim=48))


def _settings(dim: int):
    from latent_workspace.config import Settings

    return Settings(_env_file=None, embedding_dim=dim)
