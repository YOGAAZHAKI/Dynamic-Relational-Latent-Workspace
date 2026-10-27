"""Integration tests: embedder + FAISS store working together (Chunk 3).

Also pins down the project-wide rule: retrieval returns ONLY ids and
cosine scores. Relationship types (SUPPORTS/CONTRADICTS/...) must never
come from similarity — that is Chunk 4's classifier's job.
"""

from __future__ import annotations

import math

import pytest

from latent_workspace.config import Settings
from latent_workspace.embeddings import MockEmbedder, build_embedder
from latent_workspace.models import AgentRole, InformationType, InformationUnit
from latent_workspace.vector import FaissVectorStore, VectorHit, build_vector_store


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


CORPUS = [
    "Rising electricity costs pushed factory output prices higher",
    "Power tariffs increased sharply for industrial consumers",
    "The championship final was decided in a penalty shootout",
    "Goalkeepers saved two penalties in extra time",
    "New telescope images revealed a distant galaxy cluster",
    "Astronomers observed light emitted billions of years ago",
]


def _make_unit(content: str, u_type: InformationType) -> InformationUnit:
    return InformationUnit(
        content=content,
        information_type=u_type,
        agent_id=AgentRole.RESEARCHER,
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# offline end-to-end: units -> embeddings -> FAISS -> retrieval
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace():
    embedder = MockEmbedder(dim=64)
    store = FaissVectorStore(dim=64)
    units = [
        _make_unit(text, InformationType.EVIDENCE if i % 2 == 0 else InformationType.FACT)
        for i, text in enumerate(CORPUS)
    ]
    vectors = embedder.embed([u.content for u in units])
    store.add([u.id for u in units], vectors)
    return embedder, store, units


def test_retrieval_finds_topically_related_units(workspace):
    embedder, store, units = workspace
    query = "Factories faced higher electricity and power costs"
    hits = store.search(embedder.embed([query])[0], top_k=3)

    assert len(hits) == 3
    assert all(isinstance(h, VectorHit) for h in hits)
    # the two electricity units must outrank the sports/space ones
    electricity_ids = {units[0].id, units[1].id}
    top2 = {h.id for h in hits[:2]}
    assert top2 == electricity_ids
    # cosine scores are ordered best-first and bounded
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_hits_expose_no_relationship_information(workspace):
    """Similarity candidates must not masquerade as relations."""
    _, store, _ = workspace
    hits = store.search([0.1] * 64, top_k=4)
    assert hits
    for hit in hits:
        assert set(vars(hit).keys()) == {"id", "score"}
        assert not hasattr(hit, "relationship_type")
        assert not hasattr(hit, "confidence")


def test_save_load_preserves_retrieval_results(workspace, tmp_path):
    embedder, store, _ = workspace
    path = tmp_path / "ws.faiss"
    store.save(path)

    restored = FaissVectorStore.load(path)
    query = embedder.embed(["higher industrial power tariffs"])[0]
    before = store.search(query, top_k=3)
    after = restored.search(query, top_k=3)
    assert [h.as_tuple() for h in before] == [h.as_tuple() for h in after]


def test_incremental_add_after_initial_persist(workspace, tmp_path):
    """Loaded stores accept new units (this is how runs resume)."""
    embedder, store, units = workspace
    path = tmp_path / "ws.faiss"
    store.save(path)

    restored = FaissVectorStore.load(path)
    # shares many exact words with the query -> must win under the
    # mock bag-of-words embedder (which rewards token overlap)
    new_unit = _make_unit("Power costs are rising for factories", InformationType.FACT)
    added = restored.add([new_unit.id], embedder.embed([new_unit.content]))
    assert added == 1

    query = embedder.embed(["why are power costs rising"])[0]
    assert restored.search(query, top_k=1)[0].id == new_unit.id


def test_factory_pipeline_matches_settings_dim():
    s = Settings(_env_file=None, mock_mode=True, embedding_dim=32)
    embedder = build_embedder(s)
    store = build_vector_store(embedder, s)
    assert embedder.dim == store.dim == 32

    vectors = embedder.embed(["alpha beta", "alpha gamma"])
    assert store.add(["u1", "u2"], vectors) == 2
    assert store.search(vectors[0], top_k=1)[0].id == "u1"


# ---------------------------------------------------------------------------
# REAL model end-to-end (skips automatically if model/network unavailable)
# ---------------------------------------------------------------------------

def test_real_model_retrieval_end_to_end():
    try:
        from latent_workspace.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except Exception as exc:  # no torch / no network / download failed
        pytest.skip(f"real embedder unavailable: {exc}")

    s = Settings(_env_file=None)
    store = build_vector_store(embedder, s)

    ids = [f"iu_{i:04d}" for i in range(len(CORPUS))]
    store.add(ids, embedder.embed(CORPUS))

    query = "industrial electricity prices jumped"
    hits = store.search(embedder.embed([query])[0], top_k=3)
    assert len(hits) == 3
    # electricity units (positions 0,1) must beat sports (2,3) and space (4,5)
    assert set(h.id for h in hits[:2]) == {"iu_0000", "iu_0001"}
