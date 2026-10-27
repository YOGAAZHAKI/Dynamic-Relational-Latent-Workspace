"""Integration tests: KnowledgeGraph -> SubgraphRetriever -> agent (chunk 6).

Proves the pieces cooperate offline: units flow into the graph, retrieval
produces a typed/stance-labeled context, and that context really reaches
a downstream agent's prompt.
"""

from __future__ import annotations

from latent_workspace.agents import ResearcherAgent
from latent_workspace.config import Settings
from latent_workspace.embeddings import MockEmbedder
from latent_workspace.graph import KnowledgeGraph, build_workspace
from latent_workspace.llm import MockLLMClient
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
)
from latent_workspace.retrieval import SubgraphRetriever, build_retriever
from latent_workspace.relations import MockRelationDetector

UNITS = [
    ("Electricity prices rose for industrial users.",
     InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.6),
    ("Regional data confirms that industrial electricity prices increased sharply.",
     InformationType.EVIDENCE, AgentRole.RESEARCHER, 0.8),
    ("Industrial electricity prices do not rise for industrial users.",
     InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.7),
    ("Industrial electricity prices rose approximately 12% for industrial users in Q3.",
     InformationType.REFINEMENT, AgentRole.CRITIC, 0.75),
    ("The football team won the championship final.",
     InformationType.FACT, AgentRole.RESEARCHER, 0.7),
]

QUERY = "did industrial electricity prices rise for industrial users?"


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, mock_mode=True, **overrides)


def build_populated_graph(settings=None):
    ws = build_workspace(settings or make_settings())
    units = {}
    for content, u_type, agent, conf in UNITS:
        unit = InformationUnit(
            content=content, information_type=u_type, agent_id=agent,
            confidence=conf,
        )
        ws.add_unit(unit)
        units[content[:12]] = unit
    return ws, units


# ---------------------------------------------------------------------------
# end-to-end retrieval over a populated workspace
# ---------------------------------------------------------------------------


def test_retrieval_end_to_end_context_shape():
    ws, units = build_populated_graph()
    ctx = build_retriever(ws).retrieve(QUERY)

    assert ctx.task_description == QUERY
    assert ctx.anchor_id is not None
    assert len(ctx) >= 3
    # irrelevant football node never enters the context
    football = units["The football"]
    assert football.id not in ctx.unit_ids()
    assert football.id not in ctx.excluded_unit_ids  # not even reached

    for item in ctx.items:
        assert ws.has_node(item.unit.id)
        assert item.stance in {"supporting", "contradicting", "neutral"}
        for rel in item.relations:
            assert rel.direction in {"incoming", "outgoing"}
            assert 0.0 <= rel.edge.confidence <= 1.0


def test_conflicts_surface_in_prompt_block():
    ws, _units = build_populated_graph()
    ctx = SubgraphRetriever(ws).retrieve(QUERY)
    block = ctx.as_prompt_block()
    assert ctx.has_conflicts()
    assert "[CONTRADICTING]" in block
    assert "Industrial electricity prices do not rise" in block
    assert "(note: context was truncated)" not in block


def test_researcher_agent_receives_retrieved_context():
    """The retrieved subgraph must actually reach the model."""
    ws, units = build_populated_graph()
    ctx = SubgraphRetriever(ws).retrieve(QUERY)

    llm = MockLLMClient()
    agent = ResearcherAgent(llm, make_settings())
    result = agent.run("Investigate industrial electricity prices", ctx)

    assert result.error is None, result.error
    user_text = llm.calls[0]["user_text"]
    assert "Retrieved workspace context" in user_text
    assert ctx.items[0].unit.id in user_text  # best item reached the prompt
    assert "[CONTRADICTING]" in user_text  # conflict labels survive too


def test_researcher_handles_empty_graph_context():
    ws = build_workspace(make_settings())  # empty graph
    ctx = SubgraphRetriever(ws).retrieve("anything")
    llm = MockLLMClient()
    result = ResearcherAgent(llm, make_settings()).run("Investigate X", ctx)
    assert result.error is None, result.error
    assert "no relevant workspace context" in llm.calls[0]["user_text"]


# ---------------------------------------------------------------------------
# persistence: retrieval is stable across save/load
# ---------------------------------------------------------------------------


def test_retrieval_identical_after_snapshot_round_trip(tmp_path):
    ws, _units = build_populated_graph()
    snapshot = ws.save(tmp_path / "graph.json")

    restored = KnowledgeGraph.load(
        snapshot,
        embedder=MockEmbedder(384),
        detector=MockRelationDetector(),
        settings=make_settings(),
    )
    before = SubgraphRetriever(ws).retrieve(QUERY)
    after = SubgraphRetriever(restored).retrieve(QUERY)

    assert before.unit_ids() == after.unit_ids()
    assert before.anchor_id == after.anchor_id
    assert [i.stance for i in before.items] == [i.stance for i in after.items]
    assert before.contradicting_ids == after.contradicting_ids


# ---------------------------------------------------------------------------
# budget behaviour on realistic content
# ---------------------------------------------------------------------------


def test_small_budget_flags_exclusions_end_to_end():
    ws, units = build_populated_graph()
    retriever = SubgraphRetriever(ws, token_budget=80)
    ctx = retriever.retrieve(QUERY)

    assert ctx.truncated is True
    assert ctx.excluded_unit_ids
    assert len(ctx) < ws.node_count
    # excluded ids are real nodes — re-fetchable with a bigger budget
    for nid in ctx.excluded_unit_ids:
        assert ws.has_node(nid)
    wide = retriever.retrieve(QUERY, token_budget=10_000)
    assert wide.truncated is False
    assert set(ctx.unit_ids()) <= set(wide.unit_ids())


def test_settings_drive_retriever_defaults():
    ws, _units = build_populated_graph()
    tight = make_settings(retrieval_max_units=2, retrieval_expansion_depth=0)
    retriever = SubgraphRetriever(ws, tight)
    ctx = retriever.retrieve(QUERY)
    assert len(ctx) <= 2
    assert all(item.unit.id in ctx.unit_ids() for item in ctx.items)
    # every included item is a seed (depth 0 = no expansion)
    ranked = {e.node_id: e for e in SubgraphRetriever(ws, tight).rank(QUERY)}
    for item in ctx.items:
        assert ranked[item.unit.id].hop == 0
