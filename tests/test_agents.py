"""Tests for the four agents (Chunk 2).

Every test runs the REAL agent code against the offline MockLLMClient —
only the LLM is swapped, never the agent.
"""

from __future__ import annotations

import pytest

from latent_workspace.agents import build_agents
from latent_workspace.agents.critic import CriticAgent
from latent_workspace.agents.planner import PlannerAgent, PlanOutput
from latent_workspace.agents.researcher import ResearcherAgent, ResearchOutput
from latent_workspace.agents.synthesizer import SynthesizerAgent
from latent_workspace.config import Settings
from latent_workspace.llm import MockLLMClient
from latent_workspace.models import (
    AgentRole,
    ContextItem,
    ContextRelation,
    InformationType,
    InformationUnit,
    ReasoningContext,
    SubtaskStatus,
)

QUERY = "Why did global chip prices rise in 2025?"


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


@pytest.fixture
def agents(settings):
    return build_agents(MockLLMClient(), settings)


@pytest.fixture
def sample_context():
    """A tiny workspace slice, as Chunk 6 will build it."""
    unit = InformationUnit(
        content="Chip supply fell 12% in Q3 after fab shutdowns.",
        information_type=InformationType.EVIDENCE,
        agent_id=AgentRole.RESEARCHER,
        confidence=0.8,
    )
    from latent_workspace.models import InformationEdge, RelationshipType

    edge = InformationEdge(
        source_node_id=unit.id,
        target_node_id="iu_target",
        relationship_type=RelationshipType.SUPPORTS,
        confidence=0.81,
        evidence_or_rationale="mock rationale",
    )
    item = ContextItem(
        unit=unit,
        relations=[ContextRelation(edge=edge, direction="outgoing")],
    )
    return ReasoningContext(task_description="check supply claim", items=[item])


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_build_agents_creates_all_four(settings):
    agents = build_agents(MockLLMClient(), settings)
    assert {a.role for a in agents.all()} == {
        AgentRole.PLANNER,
        AgentRole.RESEARCHER,
        AgentRole.CRITIC,
        AgentRole.SYNTHESIZER,
    }
    # all share one LLM client instance
    llms = {id(a.llm) for a in agents.all()}
    assert len(llms) == 1


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_planner_returns_subtasks_and_hypotheses(agents):
    result = agents.planner.run(QUERY)
    assert result.error is None, result.error
    assert result.succeeded
    assert result.agent_id is AgentRole.PLANNER

    assert len(result.subtasks) >= 2
    for st in result.subtasks:
        assert st.description
        assert st.assigned_agent in (AgentRole.RESEARCHER, AgentRole.CRITIC)
        assert st.status is SubtaskStatus.PENDING

    assert len(result.units) >= 1
    for u in result.units:
        assert u.information_type is InformationType.HYPOTHESIS
        assert u.agent_id is AgentRole.PLANNER
        assert 0.0 <= u.confidence <= 1.0

    assert result.tokens_used > 0
    assert result.duration_ms > 0


def test_planner_rejects_subtasks_assigned_to_wrong_agents(settings):
    bad = '{"subtasks": [{"description": "x", "assigned_agent": "PLANNER"}], ' \
          '"hypotheses": [{"content": "y", "confidence": 0.5}]}'
    llm = MockLLMClient(scripted={"PLANNER": bad})
    result = PlannerAgent(llm, settings).run(QUERY)
    assert not result.succeeded
    assert "attempt" in (result.error or "").lower() or "parse" in (result.error or "").lower()
    assert result.units == [] and result.subtasks == []


def test_planner_fails_gracefully_on_llm_outage(settings):
    llm = MockLLMClient(fail_keys=["PLANNER"])
    result = PlannerAgent(llm, settings).run(QUERY)
    assert result.error is not None
    assert "LLMConnectionError" in result.error
    assert result.units == [] and result.subtasks == []
    assert result.duration_ms > 0        # timing recorded even on failure


def test_plan_schema_requires_at_least_one_subtask():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PlanOutput.model_validate({"subtasks": [], "hypotheses": [{"content": "x"}]})


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

def test_researcher_returns_typed_evidence_units(agents, sample_context):
    task = "Gather key facts about the price rise"
    result = agents.researcher.run(task, sample_context, subtask_id="st_test")
    assert result.error is None, result.error
    assert result.subtask_id == "st_test"
    assert len(result.units) >= 2
    for u in result.units:
        assert u.information_type in (InformationType.EVIDENCE, InformationType.FACT)
        assert u.agent_id is AgentRole.RESEARCHER
        assert u.metadata["subtask_id"] == "st_test"
        assert isinstance(u.metadata["sources"], list)
        assert isinstance(u.metadata["depends_on"], list)


def test_researcher_receives_workspace_context_in_prompt(settings):
    """The retrieved subgraph must actually reach the model."""
    llm = MockLLMClient()
    agent = ResearcherAgent(llm, Settings(_env_file=None, mock_mode=True))
    ctx_unit_id = "iu_abcdef12"
    ctx = ReasoningContext(
        task_description="t",
        items=[
            ContextItem(
                unit=InformationUnit(
                    id=ctx_unit_id,
                    content="A fact from the workspace.",
                    information_type=InformationType.FACT,
                    agent_id=AgentRole.RESEARCHER,
                )
            )
        ],
    )
    agent.run("Investigate X", ctx)
    user_text = llm.calls[0]["user_text"]
    assert ctx_unit_id in user_text                 # unit id is in the prompt
    assert "Retrieved workspace context" in user_text


def test_researcher_handles_empty_context_gracefully(agents):
    result = agents.researcher.run("Investigate X with no context")
    assert result.error is None, result.error
    assert result.units


def test_researcher_reports_structured_failure_on_garbage_reply(settings):
    llm = MockLLMClient(scripted={"RESEARCHER": "Sorry, no JSON today."})
    result = ResearcherAgent(llm, settings).run("Investigate X")
    assert not result.succeeded
    assert "LLMInvalidResponseError" in (result.error or "")
    assert result.units == []


def test_researcher_reports_failure_on_schema_violation(settings):
    # valid JSON, but researcher emits a HYPOTHESIS (forbidden)
    bad = '{"evidence": [{"content": "x", "information_type": "HYPOTHESIS"}]}'
    llm = MockLLMClient(scripted={"RESEARCHER": bad})
    result = ResearcherAgent(llm, settings).run("Investigate X")
    assert not result.succeeded
    assert result.units == []


def test_research_output_schema_needs_items():
    with pytest.raises(Exception):
        ResearchOutput.model_validate({"evidence": []})


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_critic_returns_targeted_critiques(agents, sample_context):
    task = "Audit the supply-shortage hypothesis"
    result = agents.critic.run(task, sample_context, subtask_id="st_2")
    assert result.error is None, result.error
    assert len(result.units) >= 1
    allowed = {
        InformationType.COUNTER_EVIDENCE,
        InformationType.REFINEMENT,
        InformationType.CONCLUSION,
        InformationType.FACT,
    }
    for u in result.units:
        assert u.information_type in allowed
        assert u.agent_id is AgentRole.CRITIC
        assert u.metadata["subtask_id"] == "st_2"
        assert "issues" in u.metadata
    # mock critic always targets something -> target recorded
    assert any("target" in u.metadata for u in result.units)


def test_critic_failure_path(settings):
    llm = MockLLMClient(fail_keys=["CRITIC"])
    result = CriticAgent(llm, settings).run("Audit X")
    assert not result.succeeded
    assert "LLMConnectionError" in (result.error or "")


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def test_synthesizer_returns_grounded_answer_with_citations(agents, sample_context):
    result = agents.synthesizer.run(QUERY, sample_context)
    assert result.error is None, result.error
    assert result.answer.strip()
    assert result.agent_id is AgentRole.SYNTHESIZER
    assert "[iu_" in result.answer            # cites unit ids
    assert result.tokens_used > 0
    assert result.duration_ms > 0


def test_synthesizer_empty_reply_is_an_error(settings):
    llm = MockLLMClient(scripted={"SYNTHESIZER": "   "})
    result = SynthesizerAgent(llm, settings).run(QUERY)
    assert not result.succeeded
    assert "empty" in (result.error or "").lower()


def test_synthesizer_failure_path(settings):
    llm = MockLLMClient(fail_keys=["SYNTHESIZER"])
    result = SynthesizerAgent(llm, settings).run(QUERY)
    assert not result.succeeded
    assert result.answer == ""


# ---------------------------------------------------------------------------
# cross-agent guarantees
# ---------------------------------------------------------------------------

def test_every_agent_survives_unexpected_crash(settings):
    """BaseAgent.run must convert ANY exception into AgentResult.error."""

    class _ExplodingLLM:
        model_name = "boom"

        def complete(self, *a, **k):
            raise ValueError("totally unexpected bug")

    llm = _ExplodingLLM()
    for agent_cls in (PlannerAgent, ResearcherAgent, CriticAgent, SynthesizerAgent):
        result = agent_cls(llm, settings).run(QUERY)
        assert result.error is not None
        assert "unexpected ValueError" in result.error
        assert result.duration_ms >= 0


def test_mock_and_real_agent_share_identical_interface(settings):
    """Agent code must not branch on mock vs real — only the client differs."""
    from latent_workspace.llm.base import LLMClient

    for agent in build_agents(MockLLMClient(), settings).all():
        assert isinstance(agent.llm, LLMClient)
