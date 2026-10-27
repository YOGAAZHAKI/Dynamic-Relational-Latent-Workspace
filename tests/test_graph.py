"""Tests for KnowledgeGraph: incremental updates, queries, failure safety.

Offline end-to-end: MockEmbedder + FaissVectorStore + MockRelationDetector.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from latent_workspace.config import Settings
from latent_workspace.embeddings import MockEmbedder
from latent_workspace.graph import (
    AddUnitResult,
    KnowledgeGraph,
    KnowledgeGraphError,
    build_workspace,
)
from latent_workspace.models import (
    AgentRole,
    InformationEdge,
    InformationType,
    InformationUnit,
    RelationshipType,
)
from latent_workspace.relations import MockRelationDetector, RelationDetectorError
from latent_workspace.vector import FaissVectorStore


def make_unit(
    content: str,
    u_type: InformationType = InformationType.EVIDENCE,
    agent: AgentRole = AgentRole.RESEARCHER,
    confidence: float = 0.7,
) -> InformationUnit:
    return InformationUnit(
        content=content,
        information_type=u_type,
        agent_id=agent,
        confidence=confidence,
    )


H = "Vaccines prevent infection."
E = "Regional data confirms vaccines prevent infection."
C = "The study shows that vaccines do not prevent infection."
WEAK = (
    # Shared-topic pair that clears the cosine floor (~0.46) so it becomes
    # a real candidate, but yields only a weak SUPPORTS prediction (~0.56)
    # that falls below relation_threshold -> must be skipped with a reason.
    "Budget allocations were finalized in March.",
    "The auditors reviewed budget numbers in March.",
)


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


@pytest.fixture
def graph(settings):
    return build_workspace(settings)


# ---------------------------------------------------------------------------
# construction / dependency injection
# ---------------------------------------------------------------------------

def test_build_workspace_injects_matching_components(graph):
    assert graph.node_count == 0 and graph.edge_count == 0
    assert graph.embedder.dim == graph.store.dim == 384
    assert isinstance(graph.detector, MockRelationDetector)


def test_constructor_rejects_dim_mismatch(settings):
    with pytest.raises(KnowledgeGraphError, match="!= vector store dim"):
        KnowledgeGraph(
            MockEmbedder(dim=64), FaissVectorStore(384),
            MockRelationDetector(), settings,
        )


def test_constructor_rejects_zero_dim(settings):
    class NoDim:
        model_name = "x"
        dim = 0

        def embed(self, texts):
            return []

    with pytest.raises(KnowledgeGraphError, match="invalid dims"):
        KnowledgeGraph(NoDim(), FaissVectorStore(384), MockRelationDetector(), settings)


# ---------------------------------------------------------------------------
# add_unit: happy path, embedding, candidates, edges
# ---------------------------------------------------------------------------

def test_first_unit_creates_node_without_edges(graph):
    h = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    result = graph.add_unit(h)

    assert isinstance(result, AddUnitResult)
    assert result.unit_id == h.id
    assert graph.node_count == 1 and graph.edge_count == 0
    assert result.candidate_ids == []          # nothing to compare against
    assert result.errors == [] and result.skipped == []
    assert h.has_embedding                     # embedded on the fly
    assert graph.store.has(h.id)               # vector indexed


def test_second_unit_builds_support_edge_with_direction(graph):
    h = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    graph.add_unit(h)
    e = make_unit(E)
    result = graph.add_unit(e)

    assert h.id in result.candidate_ids
    assert e.id not in result.candidate_ids   # the new node never sees itself
    assert len(result.edges_added) == 1

    edge = result.edges_added[0]
    assert edge.source_node_id == e.id        # newer unit is the source...
    assert edge.target_node_id == h.id        # ...pointing at the older target
    assert edge.relationship_type is RelationshipType.SUPPORTS
    assert edge.confidence >= 0.65
    assert "mock-rules" in edge.evidence_or_rationale
    assert graph.edge_count == 1


def test_contradicting_evidence_is_linked(graph):
    h = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    graph.add_unit(h)
    c = make_unit(C, InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.7)
    graph.add_unit(c)

    incoming = graph.incoming_edges(h.id, RelationshipType.CONTRADICTS)
    assert len(incoming) == 1
    assert incoming[0].source_node_id == c.id
    assert incoming[0].confidence >= 0.70


def test_unrelated_unit_creates_no_edges(graph):
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    graph.add_unit(make_unit(E))
    u = make_unit("The football team won the championship final.")
    result = graph.add_unit(u)

    assert graph.node_count == 3              # node still added...
    assert result.edges_added == []           # ...but no edges invented
    assert all(
        u.id not in (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges()
    )


def test_low_confidence_pair_is_skipped_with_reason(graph):
    graph.add_unit(make_unit(WEAK[0]))
    weak2 = make_unit(WEAK[1])
    result = graph.add_unit(weak2)

    assert result.edges_added == []
    assert any("below threshold" in reason for reason in result.skipped)
    assert graph.edge_count == 0
    # the weak pair WAS a candidate (it clears the cosine floor)
    assert result.candidate_ids, "expected the weak pair to be retrieved"


def test_identical_content_under_new_id_becomes_redundant_edge(graph):
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    twin = make_unit(H, InformationType.FACT)
    result = graph.add_unit(twin)

    types = {edge.relationship_type for edge in result.edges_added}
    assert RelationshipType.REDUNDANT in types


# ---------------------------------------------------------------------------
# duplicates and self-comparison guards
# ---------------------------------------------------------------------------

def test_duplicate_node_id_rejected(graph):
    unit = make_unit(H)
    graph.add_unit(unit)
    with pytest.raises(KnowledgeGraphError, match="Duplicate node id"):
        graph.add_unit(unit)
    assert graph.node_count == 1 and graph.store.size == 1


def test_blank_content_rejected_without_state_change(graph):
    unit = make_unit("temporary content")
    unit.content = "   "  # bypass pydantic (no validate_assignment)
    with pytest.raises(KnowledgeGraphError, match="blank content"):
        graph.add_unit(unit)
    assert graph.node_count == 0 and graph.store.size == 0


def test_wrong_type_rejected(graph):
    with pytest.raises(KnowledgeGraphError, match="expects an InformationUnit"):
        graph.add_unit("not a unit")


def test_new_node_is_never_a_candidate_of_itself(graph):
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    e = make_unit(E)
    result = graph.add_unit(e)
    assert e.id not in result.candidate_ids


# ---------------------------------------------------------------------------
# failure handling: detector / embedder errors must not corrupt state
# ---------------------------------------------------------------------------

class FlakyDetector(MockRelationDetector):
    """Fails on pairs containing the token 'BOOM'."""

    def detect(self, source_text, target_text):
        if "BOOM" in source_text or "BOOM" in target_text:
            raise RelationDetectorError("simulated detector failure")
        return super().detect(source_text, target_text)


def test_detector_failure_is_recorded_but_node_survives(settings):
    graph = KnowledgeGraph(
        MockEmbedder(384), FaissVectorStore(384), FlakyDetector(), settings
    )
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    graph.add_unit(make_unit(E))  # normal pair -> edge

    boom = make_unit("BOOM vaccines prevent infection claim")
    result = graph.add_unit(boom)

    assert any("simulated detector failure" in err for err in result.errors)
    assert graph.has_node(boom.id)             # node added despite failure
    assert graph.edge_count == 1               # healthy pair still linked


def test_detector_crash_is_recorded_not_raised(settings):
    class AngryDetector(MockRelationDetector):
        def detect(self, *args):
            raise RuntimeError("kaboom")

    graph = KnowledgeGraph(
        MockEmbedder(384), FaissVectorStore(384), AngryDetector(), settings
    )
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    result = graph.add_unit(make_unit(E))

    assert len(result.errors) == 1
    assert "detector crash RuntimeError: kaboom" in result.errors[0]
    assert result.edges_added == []
    assert graph.node_count == 2 and graph.edge_count == 0


def test_non_prediction_return_is_recorded(settings):
    class DuckDetector(MockRelationDetector):
        def detect(self, *args):
            return SimpleNamespace(  # not a RelationPrediction
                relationship_type=RelationshipType.SUPPORTS,
                confidence=9.9,
                rationale="invalid duck",
            )

    graph = KnowledgeGraph(
        MockEmbedder(384), FaissVectorStore(384), DuckDetector(), settings
    )
    graph.add_unit(make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6))
    result = graph.add_unit(make_unit(E))

    assert any("expected RelationPrediction" in err for err in result.errors)
    assert result.edges_added == []
    assert graph.edge_count == 0


def test_embedder_failure_leaves_graph_untouched(settings):
    class DeadEmbedder:
        model_name = "dead"
        dim = 384

        def embed(self, texts):
            raise RuntimeError("encoder exploded")

    graph = KnowledgeGraph(
        DeadEmbedder(), FaissVectorStore(384), MockRelationDetector(), settings
    )
    with pytest.raises(KnowledgeGraphError, match="Embedding failed"):
        graph.add_unit(make_unit(H))
    assert graph.node_count == 0 and graph.store.size == 0


def test_wrong_embedding_dim_rejected_before_commit(settings):
    class ShortEmbedder:
        model_name = "short"
        dim = 384

        def embed(self, texts):
            return [[0.1] * 16]

    graph = KnowledgeGraph(
        ShortEmbedder(), FaissVectorStore(384), MockRelationDetector(), settings
    )
    with pytest.raises(KnowledgeGraphError, match="dim"):
        graph.add_unit(make_unit(H))
    assert graph.node_count == 0 and graph.store.size == 0


# ---------------------------------------------------------------------------
# edge management: add / upsert / remove
# ---------------------------------------------------------------------------

def _two_node_graph(graph):
    u1 = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    u2 = make_unit(E)
    graph.add_unit(u1)
    graph.add_unit(u2)
    return u1, u2


def test_add_edge_requires_existing_endpoints(graph):
    u1, _u2 = _two_node_graph(graph)
    edge = InformationEdge(
        source_node_id=u1.id, target_node_id="iu_ghost",
        relationship_type=RelationshipType.SUPPORTS, confidence=0.9,
    )
    with pytest.raises(KnowledgeGraphError, match="not a node"):
        graph.add_edge(edge)


def test_add_edge_upserts_higher_confidence_only(graph):
    u1, u2 = _two_node_graph(graph)
    before = graph.edge_count

    low = InformationEdge(
        source_node_id=u1.id, target_node_id=u2.id,
        relationship_type=RelationshipType.REFINES, confidence=0.70,
    )
    stored = graph.add_edge(low)
    assert graph.edge_count == before + 1

    # higher confidence replaces (same id kept)
    high = InformationEdge(
        source_node_id=u1.id, target_node_id=u2.id,
        relationship_type=RelationshipType.REFINES, confidence=0.91,
        evidence_or_rationale="better rationale",
    )
    upgraded = graph.add_edge(high)
    assert upgraded.id == stored.id
    assert graph.edge_count == before + 1
    assert graph.get_edge(stored.id).confidence == 0.91

    # lower confidence keeps existing
    lower = InformationEdge(
        source_node_id=u1.id, target_node_id=u2.id,
        relationship_type=RelationshipType.REFINES, confidence=0.40,
    )
    kept = graph.add_edge(lower)
    assert kept.id == stored.id
    assert kept.confidence == 0.91


def test_remove_edge_and_unknown_edge(graph):
    u1, u2 = _two_node_graph(graph)
    edge = graph.outgoing_edges(u2.id)[0]  # evidence (u2) -> hypothesis (u1)
    assert edge.target_node_id == u1.id
    assert graph.remove_edge(edge.id) is True
    assert graph.remove_edge(edge.id) is False
    assert all(e.id != edge.id for e in graph.edges())


def test_remove_unit_cascades_edges_and_vector(graph):
    u1, u2 = _two_node_graph(graph)
    assert graph.edge_count >= 1

    assert graph.remove_unit(u1.id) is True
    assert not graph.has_node(u1.id)
    assert not graph.store.has(u1.id)
    assert all(
        u1.id not in (e.source_node_id, e.target_node_id)
        for e in graph.edges()
    )
    # the survivor remains fully usable
    graph.get_unit(u2.id)
    assert graph.remove_unit(u1.id) is False


# ---------------------------------------------------------------------------
# queries: supporting / contradicting / provenance
# ---------------------------------------------------------------------------

def _chain_graph():
    """C supports B supports A (two-hop grounding chain)."""
    graph = build_workspace(Settings(_env_file=None, mock_mode=True))
    a = make_unit(
        "Industrial electricity prices rose in Q3.",
        InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6,
    )
    b = make_unit(
        "Regional data confirms industrial electricity prices rose in Q3.",
        InformationType.EVIDENCE, AgentRole.RESEARCHER, 0.8,
    )
    c = make_unit(
        "Regional electricity demand data is confirmed by filings.",
        InformationType.FACT, AgentRole.RESEARCHER, 0.75,
    )
    graph.add_unit(a)
    graph.add_unit(b)
    graph.add_unit(c)
    return graph, a, b, c


def test_find_supporting_returns_units_and_edges():
    graph, a, b, c = _chain_graph()
    support = graph.find_supporting(a.id)
    assert [u.id for u, _e in support] == [b.id]
    unit, edge = support[0]
    assert unit.id == b.id
    assert edge.source_node_id == b.id and edge.target_node_id == a.id
    # C supports B, so B does have supporting evidence:
    assert [u.id for u, _e in graph.find_supporting(b.id)] == [c.id]


def test_find_contradicting_returns_counter_evidence():
    graph = build_workspace(Settings(_env_file=None, mock_mode=True))
    h = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    graph.add_unit(h)
    c = make_unit(C, InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.7)
    graph.add_unit(c)

    contra = graph.find_contradicting(h.id)
    assert [u.id for u, _e in contra] == [c.id]
    assert graph.find_contradicting(c.id) == []  # c is not itself refuted


def test_evidence_queries_respect_min_confidence():
    graph, a, b, _c = _chain_graph()
    edge = graph.find_supporting(a.id)[0][1]
    assert graph.find_supporting(a.id, min_confidence=edge.confidence) != []
    assert graph.find_supporting(a.id, min_confidence=0.99) == []


def test_queries_reject_unknown_node():
    graph, a, _b, _c = _chain_graph()
    with pytest.raises(KnowledgeGraphError, match="Unknown node"):
        graph.find_supporting("iu_missing")
    with pytest.raises(KnowledgeGraphError, match="Unknown node"):
        graph.provenance("iu_missing")


def test_provenance_walks_support_chain_in_depth_order():
    graph, a, b, c = _chain_graph()
    steps = graph.provenance(a.id)
    assert [(s.unit.id, s.depth) for s in steps] == [(b.id, 1), (c.id, 2)]
    assert steps[0].edge.source_node_id == b.id
    # grounded by support edges only
    assert all(s.edge.relationship_type in (
        RelationshipType.SUPPORTS,
        RelationshipType.DEPENDS_ON,
        RelationshipType.REFINES,
    ) for s in steps)


def test_provenance_is_cycle_safe():
    graph = build_workspace(Settings(_env_file=None, mock_mode=True))
    u1, u2 = _two_node_graph(graph)
    graph.add_edge(InformationEdge(
        source_node_id=u2.id, target_node_id=u1.id,
        relationship_type=RelationshipType.SUPPORTS, confidence=0.9,
    ))
    graph.add_edge(InformationEdge(
        source_node_id=u1.id, target_node_id=u2.id,
        relationship_type=RelationshipType.SUPPORTS, confidence=0.9,
    ))
    steps = graph.provenance(u1.id)   # u1 -> u2 -> u1 -> ... must terminate
    assert [s.unit.id for s in steps] == [u2.id]


def test_provenance_depth_limit():
    graph, a, _b, _c = _chain_graph()
    assert [s.unit.id for s in graph.provenance(a.id, max_depth=1)] == [
        graph.find_supporting(a.id)[0][0].id
    ]


def test_provenance_excludes_contradictions():
    graph = build_workspace(Settings(_env_file=None, mock_mode=True))
    h = make_unit(H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6)
    graph.add_unit(h)
    c = make_unit(C, InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.7)
    graph.add_unit(c)
    # counter-evidence exists but is NOT provenance
    assert graph.find_contradicting(h.id)
    assert graph.provenance(h.id) == []


# ---------------------------------------------------------------------------
# integrity validation
# ---------------------------------------------------------------------------

def test_validate_integrity_accepts_healthy_graph():
    graph, _a, _b, _c = _chain_graph()
    graph.validate_integrity()  # must not raise


def test_validate_integrity_detects_desynced_vector_index():
    graph, a, _b, _c = _chain_graph()
    graph._nodes["iu_ghost"] = make_unit("ghost node")  # not indexed
    graph._nodes["iu_ghost"].set_embedding([0.0] * 384)
    with pytest.raises(KnowledgeGraphError, match="out of sync"):
        graph.validate_integrity()
