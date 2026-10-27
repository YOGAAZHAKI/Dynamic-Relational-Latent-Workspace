"""Unit tests for SubgraphRetriever (chunk 6).

Covers: ranking, expansion depth, hop decay, type weights, stance
(supporting/contradicting), token budget, deterministic tie-breaking,
empty graphs, disconnected nodes and validation errors.

Everything runs offline: MockEmbedder + FAISS + MockRelationDetector.
"""

from __future__ import annotations

import pytest

from latent_workspace.config import Settings
from latent_workspace.graph import build_workspace
from latent_workspace.models import (
    AgentRole,
    InformationEdge,
    InformationType,
    InformationUnit,
    RelationshipType,
)
from latent_workspace.retrieval import (
    RetrievalError,
    SubgraphRetriever,
    build_retriever,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def make_unit(
    content: str,
    u_type=InformationType.EVIDENCE,
    agent=AgentRole.RESEARCHER,
    conf: float = 0.8,
    uid: str | None = None,
) -> InformationUnit:
    kwargs = {"id": uid} if uid else {}
    return InformationUnit(
        content=content,
        information_type=u_type,
        agent_id=agent,
        confidence=conf,
        **kwargs,
    )


V_H = "Vaccines prevent infection."
V_E = "Regional data confirms vaccines prevent infection."
V_C = "The study shows that vaccines do not prevent infection."


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


@pytest.fixture
def graph(settings):
    return build_workspace(settings)


def add_vaccine_trio(graph) -> dict[str, InformationUnit]:
    """H <-supported-by E; H <-contradicted-by C."""
    units = {
        "h": make_unit(V_H, InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6),
        "e": make_unit(V_E),
        "c": make_unit(V_C, InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.7),
    }
    for key in ("h", "e", "c"):
        graph.add_unit(units[key])
    return units


def edge(source: str, target: str, rtype: RelationshipType, conf: float):
    return InformationEdge(
        source_node_id=source,
        target_node_id=target,
        relationship_type=rtype,
        confidence=conf,
    )


def chain_graph(settings=None):
    """Seed S <- B <- D via manual SUPPORTS edges (disjoint vocab).

    Only S shares words with queries, so seeds are fully controlled:
    B and D are reachable only by graph expansion.
    """
    ws = build_workspace(settings or Settings(_env_file=None, mock_mode=True))
    s = make_unit("Alpha mission launch window closed early.", uid="iu_seed001")
    b = make_unit("Gamma protocol requires calibration.", uid="iu_mid0002")
    d = make_unit("Delta orbit was confirmed twice.", uid="iu_far0003")
    ws.add_unit(s)
    ws.add_unit(b)
    ws.add_unit(d)
    ws.add_edge(edge(b.id, s.id, RelationshipType.SUPPORTS, 0.8))
    ws.add_edge(edge(d.id, b.id, RelationshipType.SUPPORTS, 0.8))
    return ws, s, b, d


QUERY = "alpha mission launch"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefgh") == 2  # 8 chars / 4
    assert estimate_tokens("x") == 1  # never zero for non-empty


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------


def test_retriever_requires_a_graph(graph):
    with pytest.raises(RetrievalError, match="KnowledgeGraph"):
        SubgraphRetriever(object())


def test_unknown_type_weight_rejected(graph):
    with pytest.raises(RetrievalError, match="Unknown relationship type"):
        SubgraphRetriever(graph, type_weights={"BOGUS": 1.0})


def test_unrelated_type_weight_rejected(graph):
    with pytest.raises(RetrievalError, match="UNRELATED"):
        SubgraphRetriever(graph, type_weights={"UNRELATED": 0.5})


def test_negative_depth_rejected(graph):
    with pytest.raises(RetrievalError, match="expansion_depth"):
        SubgraphRetriever(graph, expansion_depth=-1)


def test_min_edge_confidence_bounds(graph):
    with pytest.raises(RetrievalError, match="min_edge_confidence"):
        SubgraphRetriever(graph, min_edge_confidence=1.5)


def test_blank_query_rejected(graph):
    retriever = SubgraphRetriever(graph)
    with pytest.raises(RetrievalError, match="non-blank"):
        retriever.retrieve("   ")
    with pytest.raises(RetrievalError, match="non-blank"):
        retriever.retrieve(None)  # type: ignore[arg-type]


def test_build_retriever_factory(graph, settings):
    retriever = build_retriever(graph, settings, expansion_depth=3)
    assert isinstance(retriever, SubgraphRetriever)
    assert retriever.expansion_depth == 3


def test_seed_k_widens_the_semantic_seed_row(settings):
    """Callers may opt into extra recall without changing the store top_k."""
    graph = build_workspace(settings)
    for i in range(3):
        unit = make_unit(f"alpha mission launch review {i}", uid=f"iu_seed{i:04d}")
        graph.add_unit(unit)

    narrow = SubgraphRetriever(graph, seed_k=1).rank("alpha mission launch")
    wide = SubgraphRetriever(graph, seed_k=3).rank("alpha mission launch")
    assert len(narrow) == 1
    assert len(wide) == 3


def test_negative_seed_k_rejected(graph):
    with pytest.raises(RetrievalError, match="seed_k"):
        SubgraphRetriever(graph, seed_k=-1)


# ---------------------------------------------------------------------------
# empty graphs and irrelevant queries
# ---------------------------------------------------------------------------


def test_empty_graph_returns_empty_context(graph):
    ctx = SubgraphRetriever(graph).retrieve("anything at all")
    assert len(ctx) == 0
    assert ctx.anchor_id is None
    assert ctx.token_estimate == 0
    assert ctx.truncated is False
    assert "no relevant workspace context" in ctx.as_prompt_block()


def test_rank_on_empty_graph_returns_nothing(graph):
    assert SubgraphRetriever(graph).rank("anything") == []


def test_irrelevant_node_is_never_seeded(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    football = make_unit("The football team won the championship final.")
    graph.add_unit(football)

    retriever = SubgraphRetriever(graph)
    ranked_ids = [s.node_id for s in retriever.rank("vaccines prevent infection")]
    assert football.id not in ranked_ids
    assert set(ranked_ids) == {units["h"].id, units["e"].id, units["c"].id}


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_best_semantic_match_is_top_ranked(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ranked = SubgraphRetriever(graph).rank("vaccines prevent infection")
    # the hypothesis is the verbatim query -> cosine 1.0 -> rank 1
    assert ranked[0].node_id == units["h"].id
    assert ranked[0].hop == 0


def test_rank_reports_score_breakdown(settings):
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    ranked = SubgraphRetriever(graph).rank("vaccines prevent infection")
    for entry in ranked:
        assert entry.score == pytest.approx(entry.semantic + entry.relational, abs=1e-4)
        assert entry.hop >= 0


def test_score_components_are_separated():
    """Seeds earn semantic score; expanded neighbors earn relational.

    (A seed can *also* earn relational signal from seed-to-seed edges —
    that is intentional — so the clean separation is checked on the
    controlled chain graph where vocab is disjoint.)
    """
    ws, s, b, d = chain_graph()
    ranked = {e.node_id: e for e in SubgraphRetriever(ws, expansion_depth=2).rank(QUERY)}
    # the seed matches the query semantically and has no connected seeds
    assert ranked[s.id].semantic > 0.0 and ranked[s.id].relational == 0.0
    # neighbors share no vocabulary (semantic ~0) and are reached only
    # through edges (relational > 0, decaying with hop)
    assert ranked[b.id].semantic == 0.0 and ranked[b.id].relational > 0.0
    assert ranked[d.id].semantic == 0.0 and ranked[d.id].relational > 0.0


# ---------------------------------------------------------------------------
# graph expansion
# ---------------------------------------------------------------------------


def test_depth_zero_returns_seeds_only():
    ws, s, _b, _d = chain_graph()
    retriever = SubgraphRetriever(ws, expansion_depth=0)
    ctx = retriever.retrieve(QUERY)
    assert ctx.unit_ids() == [s.id]


def test_depth_one_includes_direct_neighbors():
    ws, s, b, _d = chain_graph()
    retriever = SubgraphRetriever(ws, expansion_depth=1)
    ids = set(retriever.retrieve(QUERY).unit_ids())
    assert ids == {s.id, b.id}


def test_depth_two_reaches_second_hop():
    ws, s, b, d = chain_graph()
    retriever = SubgraphRetriever(ws, expansion_depth=2)
    ids = set(retriever.retrieve(QUERY).unit_ids())
    assert ids == {s.id, b.id, d.id}


def test_hop_decay_shrinks_relational_signal():
    ws, s, b, d = chain_graph()
    ranked = {e.node_id: e for e in SubgraphRetriever(ws, expansion_depth=2).rank(QUERY)}
    # identical edge confidences; the farther hop is decayed by hop_decay
    assert ranked[b.id].hop == 1 and ranked[d.id].hop == 2
    assert ranked[b.id].relational == pytest.approx(0.4 * 0.8, abs=1e-3)
    assert ranked[d.id].relational == pytest.approx(0.4 * 0.8 * 0.5, abs=1e-3)
    assert ranked[d.id].relational < ranked[b.id].relational


def test_min_edge_confidence_blocks_expansion():
    ws, s, b, d = chain_graph()
    retriever = SubgraphRetriever(
        ws, expansion_depth=2, min_edge_confidence=0.85  # edges are 0.8
    )
    ids = set(retriever.retrieve(QUERY).unit_ids())
    assert ids == {s.id}  # neighbors exist but their edges are too weak


def test_min_edge_confidence_hides_weak_relations(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)  # E supports H at ~0.77
    retriever = SubgraphRetriever(graph, min_edge_confidence=0.99)
    ctx = retriever.retrieve("vaccines prevent infection")
    h_item = next(i for i in ctx.items if i.unit.id == units["h"].id)
    assert h_item.relations == []


# ---------------------------------------------------------------------------
# relationship-type priorities
# ---------------------------------------------------------------------------


def boost_graph(settings=None):
    """Anchor S with a SUPPORTS neighbor and a CONTRADICTS neighbor."""
    ws = build_workspace(settings or Settings(_env_file=None, mock_mode=True))
    a = make_unit("Alpha mission launch window closed early.", uid="iu_anchor1")
    supp = make_unit("Gamma protocol requires calibration.", uid="iu_sup0002")
    contra = make_unit("Delta orbit was confirmed twice.", uid="iu_con0003")
    ws.add_unit(a)
    ws.add_unit(supp)
    ws.add_unit(contra)
    ws.add_edge(edge(supp.id, a.id, RelationshipType.SUPPORTS, 0.8))
    ws.add_edge(edge(contra.id, a.id, RelationshipType.CONTRADICTS, 0.8))
    return ws, a, supp, contra


def test_type_weights_scale_relational_component():
    ws, a, supp, contra = boost_graph()
    ranked = {
        e.node_id: e
        for e in SubgraphRetriever(
            ws, expansion_depth=1, type_weights={"SUPPORTS": 0.25}
        ).rank(QUERY)
    }
    # identical edge confidence, different type weight -> different signal
    assert ranked[supp.id].relational == pytest.approx(0.4 * 0.8 * 0.25, abs=1e-3)
    assert ranked[contra.id].relational == pytest.approx(0.4 * 0.8 * 1.0, abs=1e-3)


def test_contradicts_weight_matches_supports_by_default():
    ws, a, supp, contra = boost_graph()
    ranked = {
        e.node_id: e for e in SubgraphRetriever(ws, expansion_depth=1).rank(QUERY)
    }
    # default weights: CONTRADICTS == SUPPORTS (conflicts must surface)
    assert ranked[supp.id].relational == pytest.approx(
        ranked[contra.id].relational, abs=1e-9
    )


# ---------------------------------------------------------------------------
# stance: supporting vs contradicting evidence
# ---------------------------------------------------------------------------


def test_stances_against_anchor(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection")

    assert ctx.anchor_id == units["h"].id  # verbatim query match
    by_id = {i.unit.id: i for i in ctx.items}
    assert by_id[units["e"].id].stance == "supporting"
    assert by_id[units["c"].id].stance == "contradicting"
    assert by_id[units["h"].id].stance == "neutral"  # anchor is the focus
    assert ctx.supporting_ids == [units["e"].id]
    assert ctx.contradicting_ids == [units["c"].id]
    assert ctx.has_conflicts() is True


def test_conflict_wins_over_support_for_stance(graph):
    a = make_unit("Alpha mission launch window closed early.", uid="iu_anchor1")
    b = make_unit("Gamma protocol requires calibration.", uid="iu_both002")
    graph.add_unit(a)
    graph.add_unit(b)
    graph.add_edge(edge(b.id, a.id, RelationshipType.SUPPORTS, 0.8))
    graph.add_edge(edge(b.id, a.id, RelationshipType.CONTRADICTS, 0.8))

    ctx = SubgraphRetriever(graph, expansion_depth=1).retrieve(QUERY)
    b_item = next(i for i in ctx.items if i.unit.id == b.id)
    assert b_item.stance == "contradicting"


def test_contradicting_evidence_is_retrieved_not_dropped(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection")
    assert units["c"].id in ctx.unit_ids()  # counter-evidence survived
    block = ctx.as_prompt_block()
    assert "[CONTRADICTING]" in block


def test_budget_cut_marks_excluded_conflicts(settings):
    """A tiny budget must still FLAG dropped conflicts, not hide them."""
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve(
        "vaccines prevent infection", token_budget=60, max_units=10
    )
    assert ctx.truncated is True
    assert len(ctx) < 3
    # the anchor stayed; both other units are recorded as excluded
    assert ctx.anchor_id in ctx.unit_ids()
    assert set(ctx.excluded_unit_ids) == {units["e"].id, units["c"].id}


def test_explicit_anchor_id_overrides_best_seed(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve(
        "vaccines prevent infection", anchor_id=units["e"].id
    )
    assert ctx.anchor_id == units["e"].id
    by_id = {i.unit.id: i for i in ctx.items}
    assert by_id[units["h"].id].stance == "supporting"  # H is supported BY E
    assert by_id[units["c"].id].stance == "contradicting"  # C contradicts E too


def test_unknown_anchor_rejected(settings):
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    with pytest.raises(RetrievalError, match="anchor_id"):
        SubgraphRetriever(graph).retrieve("vaccines", anchor_id="iu_nope000")


# ---------------------------------------------------------------------------
# relations + provenance on items
# ---------------------------------------------------------------------------


def test_item_relations_are_typed_and_confidence_labeled(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection")
    h_item = next(i for i in ctx.items if i.unit.id == units["h"].id)
    labels = h_item.relation_labels
    assert labels, "anchor H should carry its incident edges"
    assert any(label.startswith("SUPPORTS(") for label in labels)
    assert any(label.startswith("CONTRADICTS(") for label in labels)
    assert all("conf=" in label for label in labels)


def test_provenance_refs_attached_where_chain_exists(settings):
    graph = build_workspace(settings)
    units = add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection")
    h_item = next(i for i in ctx.items if i.unit.id == units["h"].id)
    assert h_item.provenance, "H is grounded by E -> provenance expected"
    first = h_item.provenance[0]
    assert first.unit_id == units["e"].id
    assert first.relationship_type is RelationshipType.SUPPORTS
    assert first.depth == 1


def test_disconnected_query_match_is_included(settings):
    """A node with NO edges but a strong query match still comes back."""
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    football = make_unit("The football team won the championship final.")
    graph.add_unit(football)

    ctx = SubgraphRetriever(graph).retrieve("football championship team won")
    f_item = next(i for i in ctx.items if i.unit.id == football.id)
    assert f_item.relations == []  # nothing to connect
    assert f_item.stance == "neutral"
    assert f_item.unit.id in ctx.as_prompt_block()


# ---------------------------------------------------------------------------
# budgets and caps
# ---------------------------------------------------------------------------


def test_token_budget_truncates_and_excludes(settings):
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve(
        "vaccines prevent infection", token_budget=50
    )
    assert ctx.truncated is True
    assert ctx.excluded_unit_ids
    assert ctx.token_estimate > 0
    assert "(note: context was truncated)" in ctx.as_prompt_block()


def test_max_units_cap(settings):
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection", max_units=1)
    assert len(ctx) == 1
    assert ctx.truncated is True
    assert len(ctx.excluded_unit_ids) == 2


def test_first_item_survives_impossible_budget(settings):
    """Even a zero-ish budget keeps one item rather than nothing."""
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    ctx = SubgraphRetriever(graph).retrieve("vaccines prevent infection", token_budget=0)
    assert len(ctx) == 1
    assert ctx.truncated is True


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_tied_scores_break_by_unit_id(settings):
    graph = build_workspace(settings)
    a = make_unit("Shared identical statement.", uid="iu_aaaaaaaa")
    b = make_unit("Shared identical statement.", uid="iu_bbbbbbbb")
    graph.add_unit(a)
    graph.add_unit(b)

    ranked = SubgraphRetriever(graph).rank("shared identical statement")
    tied = [e for e in ranked if e.node_id in {a.id, b.id}]
    assert [e.node_id for e in tied] == [a.id, b.id]  # id ascending
    assert tied[0].score == tied[1].score


def test_repeated_retrieval_is_identical(settings):
    graph = build_workspace(settings)
    add_vaccine_trio(graph)
    retriever = SubgraphRetriever(graph)
    first = retriever.retrieve("vaccines prevent infection")
    second = retriever.retrieve("vaccines prevent infection")
    assert first.unit_ids() == second.unit_ids()
    assert [i.stance for i in first.items] == [i.stance for i in second.items]
    assert first.anchor_id == second.anchor_id
