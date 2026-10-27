"""Tests for KnowledgeGraph snapshot persistence + restoration (Chunk 5).

Covers: round trips, incremental adds after reload, and every corruption
mode (bad JSON, wrong format/version, dangling edges, duplicates,
self-edges, UNRELATED edges, dim mismatches, vector-index desync).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latent_workspace.config import Settings
from latent_workspace.embeddings import MockEmbedder
from latent_workspace.graph import KnowledgeGraph, KnowledgeGraphError, build_workspace
from latent_workspace.graph.workspace import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_VERSION,
)
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
)
from latent_workspace.relations import MockRelationDetector


def make_unit(content: str, u_type=InformationType.EVIDENCE) -> InformationUnit:
    return InformationUnit(
        content=content,
        information_type=u_type,
        agent_id=AgentRole.RESEARCHER,
        confidence=0.75,
    )


H = "Vaccines prevent infection."
E = "Regional data confirms vaccines prevent infection."
C = "The study shows that vaccines do not prevent infection."


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


@pytest.fixture
def graph(settings):
    ws = build_workspace(settings)
    ws.add_unit(make_unit(H, InformationType.HYPOTHESIS))
    ws.add_unit(make_unit(E))
    ws.add_unit(make_unit(C, InformationType.COUNTER_EVIDENCE))
    return ws


def snapshot_path(tmp_path: Path) -> Path:
    return tmp_path / "graph.json"


def _load(path, settings, dim: int = 384):
    return KnowledgeGraph.load(
        path,
        embedder=MockEmbedder(dim),
        detector=MockRelationDetector(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

def test_save_creates_snapshot_and_vector_index(graph, tmp_path):
    path = graph.save(snapshot_path(tmp_path))
    assert path.exists()
    assert path.with_suffix(".faiss").exists()
    assert Path(str(path.with_suffix(".faiss")) + ".ids.json").exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == SNAPSHOT_FORMAT
    assert payload["version"] == SNAPSHOT_VERSION
    assert payload["embedding_dim"] == 384
    assert len(payload["nodes"]) == graph.node_count
    assert len(payload["edges"]) == graph.edge_count


def test_round_trip_preserves_structure_and_queries(graph, settings, tmp_path):
    graph.save(snapshot_path(tmp_path))
    restored = _load(snapshot_path(tmp_path), settings)

    assert restored.node_count == graph.node_count
    assert restored.edge_count == graph.edge_count
    assert {u.id for u in restored.nodes()} == {u.id for u in graph.nodes()}
    assert {e.id for e in restored.edges()} == {e.id for e in graph.edges()}

    h_id = next(u.id for u in restored.nodes() if u.content == H)
    orig_h = next(u.id for u in graph.nodes() if u.content == H)
    assert [u.id for u, _e in restored.find_supporting(h_id)] == [
        u.id for u, _e in graph.find_supporting(orig_h)
    ]
    assert [u.id for u, _e in restored.find_contradicting(h_id)] == [
        u.id for u, _e in graph.find_contradicting(orig_h)
    ]
    assert [s.unit.id for s in restored.provenance(h_id)] == [
        s.unit.id for s in graph.provenance(orig_h)
    ]


def test_round_trip_preserves_vector_search(graph, settings, tmp_path):
    graph.save(snapshot_path(tmp_path))
    restored = _load(snapshot_path(tmp_path), settings)

    unit = restored.get_unit(next(u.id for u in restored.nodes() if u.content == E))
    hits = restored.store.search(unit.embedding, top_k=3)
    assert hits and hits[0].id == unit.id  # self-match with cosine ~1
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_incremental_add_after_reload(graph, settings, tmp_path):
    graph.save(snapshot_path(tmp_path))
    restored = _load(snapshot_path(tmp_path), settings)

    fresh = make_unit("Regional data confirms vaccines prevent infection again.")
    result = restored.add_unit(fresh)
    assert restored.has_node(fresh.id)
    assert result.edges_added  # still classifying against restored candidates

    # and the re-saved snapshot loads cleanly too
    second_path = tmp_path / "graph2.json"
    restored.save(second_path)
    reloaded = _load(second_path, settings)
    assert reloaded.node_count == restored.node_count


def test_embedder_dim_mismatch_on_load(graph, settings, tmp_path):
    graph.save(snapshot_path(tmp_path))
    with pytest.raises(KnowledgeGraphError, match="does not match embedder dim"):
        _load(snapshot_path(tmp_path), settings, dim=64)


# ---------------------------------------------------------------------------
# corruption: snapshot content
# ---------------------------------------------------------------------------

def _write_tampered(graph, tmp_path, mutate):
    """Save, then tamper the JSON payload, returning the path."""
    path = graph.save(snapshot_path(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_missing_snapshot(graph, settings, tmp_path):
    with pytest.raises(KnowledgeGraphError, match="not found"):
        _load(tmp_path / "nope.json", settings)


def test_load_corrupt_json(graph, settings, tmp_path):
    path = snapshot_path(tmp_path)
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(KnowledgeGraphError, match="Corrupt"):
        _load(path, settings)


def test_load_unknown_format(graph, settings, tmp_path):
    path = _write_tampered(graph, tmp_path, lambda p: p.update(format="something"))
    with pytest.raises(KnowledgeGraphError, match="format"):
        _load(path, settings)


def test_load_unsupported_version(graph, settings, tmp_path):
    path = _write_tampered(graph, tmp_path, lambda p: p.update(version=999))
    with pytest.raises(KnowledgeGraphError, match="version"):
        _load(path, settings)


def test_load_edge_referencing_missing_node(graph, settings, tmp_path):
    def drop_a_node(payload):
        removed = payload["nodes"].pop()["id"]
        payload["nodes"] = [n for n in payload["nodes"] if n["id"] != removed]

    path = _write_tampered(graph, tmp_path, drop_a_node)
    with pytest.raises(KnowledgeGraphError, match="unknown node"):
        _load(path, settings)


def test_load_duplicate_node_ids(graph, settings, tmp_path):
    def duplicate_node(payload):
        payload["nodes"].append(dict(payload["nodes"][0]))

    path = _write_tampered(graph, tmp_path, duplicate_node)
    with pytest.raises(KnowledgeGraphError, match="Duplicate node"):
        _load(path, settings)


def test_load_rejects_self_edge(graph, settings, tmp_path):
    def make_self_edge(payload):
        payload["edges"][0]["target_node_id"] = payload["edges"][0]["source_node_id"]

    path = _write_tampered(graph, tmp_path, make_self_edge)
    with pytest.raises(KnowledgeGraphError, match="Invalid edge"):
        _load(path, settings)


def test_load_rejects_unrelated_edge(graph, settings, tmp_path):
    def make_unrelated(payload):
        payload["edges"][0]["relationship_type"] = "UNRELATED"

    path = _write_tampered(graph, tmp_path, make_unrelated)
    with pytest.raises(KnowledgeGraphError, match="Invalid edge"):
        _load(path, settings)


def test_load_rejects_out_of_range_edge_confidence(graph, settings, tmp_path):
    def boost(payload):
        payload["edges"][0]["confidence"] = 1.5

    path = _write_tampered(graph, tmp_path, boost)
    with pytest.raises(KnowledgeGraphError, match="Invalid edge"):
        _load(path, settings)


def test_load_rejects_node_embedding_dim_mismatch(graph, settings, tmp_path):
    def shrink_embedding(payload):
        payload["nodes"][0]["embedding"] = [0.1, 0.2, 0.3]

    path = _write_tampered(graph, tmp_path, shrink_embedding)
    with pytest.raises(KnowledgeGraphError, match="embedding dim"):
        _load(path, settings)


def test_load_rejects_blank_node_content(graph, settings, tmp_path):
    def blank(payload):
        payload["nodes"][0]["content"] = "   "

    path = _write_tampered(graph, tmp_path, blank)
    with pytest.raises(KnowledgeGraphError, match="Invalid node"):
        _load(path, settings)


# ---------------------------------------------------------------------------
# corruption: vector index / sync
# ---------------------------------------------------------------------------

def test_load_missing_vector_index(graph, settings, tmp_path):
    path = graph.save(snapshot_path(tmp_path))
    path.with_suffix(".faiss").unlink()
    with pytest.raises(KnowledgeGraphError, match="vector index"):
        _load(path, settings)


def test_load_missing_ids_sidecar(graph, settings, tmp_path):
    path = graph.save(snapshot_path(tmp_path))
    Path(str(path.with_suffix(".faiss")) + ".ids.json").unlink()
    with pytest.raises(KnowledgeGraphError, match="vector index"):
        _load(path, settings)


def test_load_detects_vector_node_desync(graph, settings, tmp_path):
    """Node stripped of its embedding while the index still holds it."""

    def strip_embedding(payload):
        payload["nodes"][0]["embedding"] = None

    path = _write_tampered(graph, tmp_path, strip_embedding)
    with pytest.raises(KnowledgeGraphError, match="out of sync"):
        _load(path, settings)


# ---------------------------------------------------------------------------
# save-time safety
# ---------------------------------------------------------------------------

def test_save_refuses_desynced_graph(graph, tmp_path):
    ghost = make_unit("ghost")
    ghost.set_embedding([0.0] * 384)
    graph._nodes[ghost.id] = ghost  # bypass add_unit
    with pytest.raises(KnowledgeGraphError, match="out of sync"):
        graph.save(snapshot_path(tmp_path))
    assert not snapshot_path(tmp_path).exists()


def test_save_creates_missing_directories(graph, tmp_path):
    path = graph.save(tmp_path / "deep" / "nested" / "graph.json")
    assert path.exists()


def test_snapshot_text_is_plain_json(graph, tmp_path):
    path = graph.save(snapshot_path(tmp_path))
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # parses
