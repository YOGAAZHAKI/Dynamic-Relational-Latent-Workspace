"""Tests for the end-to-end orchestrator (chunk 7).

Everything runs offline against MockLLMClient + MockEmbedder + FAISS +
MockRelationDetector. The point is to prove the *wiring*: planning feeds
the graph, workers receive retrieved subgraphs, every produced unit is
ingested, and a grounded answer comes back even when parts fail.
"""

from __future__ import annotations

import pytest

from latent_workspace.agents import build_agents
from latent_workspace.config import Settings
from latent_workspace.graph import build_workspace
from latent_workspace.llm import MockLLMClient
from latent_workspace.models import SubtaskStatus
from latent_workspace.orchestrator import (
    Orchestrator,
    OrchestratorError,
    build_orchestrator,
    extract_citations,
    fallback_answer,
)

QUERY = "Why did global chip prices rise in 2025?"


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


def make_orchestrator(settings, llm=None, **kwargs):
    llm = llm or MockLLMClient()
    graph = build_workspace(settings)
    agents = build_agents(llm, settings)
    return Orchestrator(graph=graph, agents=agents, settings=settings, **kwargs)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_full_run_produces_grounded_answer_and_populates_graph(settings):
    orchestrator = make_orchestrator(settings)
    result = orchestrator.run(QUERY)

    assert result.errors == []
    assert result.final_answer.strip()
    assert "[iu_" in result.final_answer
    assert result.metrics.agent_calls >= 3  # planner + workers + synthesizer
    assert result.metrics.nodes_added == len(result.units)
    assert result.metrics.edges_added > 0
    assert result.metrics.rounds == 1
    assert result.contexts, "every agent context must be auditable"
    assert orchestrator.graph.node_count == len(result.units)

    # every unit reported is really in the workspace
    for unit in result.units:
        assert orchestrator.graph.has_node(unit.id)

    # every edge reported is really in the workspace
    graph_edge_ids = {edge.id for edge in orchestrator.graph.edges()}
    assert {edge.id for edge in result.edges} <= graph_edge_ids

    # citations point at real units
    citations = extract_citations(result.final_answer)
    assert citations
    for cited in citations:
        assert orchestrator.graph.has_node(cited)
    assert result.extra["answer_grounded"] is True
    assert result.extra["mode"] == "mock"


def test_all_subtasks_reach_a_terminal_status(settings):
    result = make_orchestrator(settings).run(QUERY)
    assert result.subtasks
    assert all(st.status is not SubtaskStatus.PENDING for st in result.subtasks)
    assert all(st.unit_ids for st in result.subtasks)


def test_worker_coverage_is_enforced(settings):
    """A planner that forgot the critic still gets a critic pass."""

    class ResearcherOnlyPlanner(MockLLMClient):
        def _planner(self, query):  # noqa: D401 - test double
            return (
                '{"subtasks": [{"description": "gather facts about '
                f'{query}", "assigned_agent": "RESEARCHER"}}], '
                '"hypotheses": [{"content": "a hypothesis", '
                '"confidence": 0.5}]}'
            )

    orchestrator = make_orchestrator(settings, llm=ResearcherOnlyPlanner())
    result = orchestrator.run(QUERY)
    roles = [st.assigned_agent.value for st in result.subtasks]
    assert "RESEARCHER" in roles and "CRITIC" in roles


def test_zero_rounds_skips_workers_but_still_answers(settings):
    result = make_orchestrator(settings).run(QUERY, max_rounds=0)
    assert result.metrics.rounds == 0
    assert all(st.status is SubtaskStatus.PENDING for st in result.subtasks)
    assert result.final_answer.strip()
    assert result.metrics.agent_calls == 2  # planner + synthesizer only


# ---------------------------------------------------------------------------
# failure handling: one broken part never crashes the run
# ---------------------------------------------------------------------------

def test_planner_failure_falls_back_to_one_research_task(settings):
    llm = MockLLMClient(fail_keys=["PLANNER"])
    result = make_orchestrator(settings, llm=llm).run(QUERY)

    assert any(err.startswith("PLANNER:") for err in result.errors)
    assert len(result.subtasks) == 1
    assert result.extra["planner_failed"] is True
    assert result.final_answer.strip()
    assert result.units  # the researcher still produced evidence


def test_worker_failure_marks_subtask_failed_without_aborting(settings):
    llm = MockLLMClient(fail_keys=["RESEARCHER"])
    result = make_orchestrator(settings, llm=llm).run(QUERY)

    researcher_tasks = [
        st for st in result.subtasks if st.assigned_agent.value == "RESEARCHER"
    ]
    assert researcher_tasks and all(
        st.status is SubtaskStatus.FAILED for st in researcher_tasks
    )
    assert any("RESEARCHER" in err for err in result.errors)
    # critic + synthesizer still ran
    assert result.final_answer.strip()


def test_synthesizer_failure_uses_cited_extractive_fallback(settings):
    llm = MockLLMClient(fail_keys=["SYNTHESIZER"])
    result = make_orchestrator(settings, llm=llm).run(QUERY)

    assert result.extra["synthesizer_failed"] is True
    assert any(err.startswith("SYNTHESIZER:") for err in result.errors)
    assert "extractive fallback" in result.final_answer
    citations = extract_citations(result.final_answer)
    assert citations and result.extra["answer_grounded"] is True


def test_retrieval_failure_is_recorded_and_run_continues(settings):
    class BrokenRetriever:
        def retrieve(self, query, **kwargs):
            raise RuntimeError("index unavailable")

    result = make_orchestrator(
        settings, retriever=BrokenRetriever()
    ).run(QUERY)

    assert any("retrieval failed" in err for err in result.errors)
    assert result.final_answer.strip()
    assert result.metrics.agent_calls >= 2


def test_blank_query_is_rejected(settings):
    orchestrator = make_orchestrator(settings)
    with pytest.raises(OrchestratorError, match="non-blank"):
        orchestrator.run("   ")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_extract_citations_is_unique_and_ordered():
    text = "See [iu_aaaaaaaa] and iu_bbbbbbbb, again [iu_aaaaaaaa]."
    assert extract_citations(text) == ["iu_aaaaaaaa", "iu_bbbbbbbb"]
    assert extract_citations("") == []


def test_fallback_answer_without_context_is_honest():
    from latent_workspace.models import ReasoningContext

    answer = fallback_answer("unknown question", ReasoningContext(task_description="x"))
    assert "could not find enough evidence" in answer


def test_build_orchestrator_factory_wires_shared_llm(settings):
    llm = MockLLMClient()
    orchestrator = build_orchestrator(settings, llm=llm)
    assert orchestrator.agents.planner.llm is llm
    assert orchestrator.llm is llm
    result = orchestrator.run(QUERY)
    assert result.final_answer
