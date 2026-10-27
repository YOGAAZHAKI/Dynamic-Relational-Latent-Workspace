"""Tests for the data-model layer (Chunk 1).

Run:  pytest
"""

from __future__ import annotations

from datetime import timezone

import pytest
from pydantic import ValidationError

from latent_workspace.config import Settings, get_settings
from latent_workspace.models import (
    AgentRole,
    InformationEdge,
    InformationType,
    InformationUnit,
    RelationshipType,
    Subtask,
    SubtaskStatus,
    new_id,
    utc_now,
)


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------

def test_relationship_types_match_spec():
    values = {t.value for t in RelationshipType}
    assert values == {
        "SUPPORTS",
        "CONTRADICTS",
        "REFINES",
        "DEPENDS_ON",
        "REDUNDANT",
        "UNRELATED",
    }


def test_stored_types_exclude_unrelated():
    assert RelationshipType.UNRELATED not in RelationshipType.stored_types()
    assert len(RelationshipType.stored_types()) == 5


def test_agent_roles_and_information_types():
    assert {r.value for r in AgentRole} == {"PLANNER", "RESEARCHER", "CRITIC", "SYNTHESIZER"}
    assert "HYPOTHESIS" in {t.value for t in InformationType}
    assert "COUNTER_EVIDENCE" in {t.value for t in InformationType}


# ---------------------------------------------------------------------------
# InformationUnit
# ---------------------------------------------------------------------------

def test_unit_defaults_are_assigned(unit):
    assert unit.id.startswith("iu_")
    assert unit.embedding is None          # not vectorized yet
    assert unit.timestamp.tzinfo is not None  # timezone-aware
    assert unit.metadata == {"subtask_id": "st_0001"}
    assert unit.has_embedding is False


def test_units_get_unique_ids():
    ids = {new_id("iu") for _ in range(200)}
    assert len(ids) == 200


def test_unit_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        InformationUnit(
            content="x",
            information_type=InformationType.FACT,
            agent_id=AgentRole.CRITIC,
            confidence=1.4,
        )
    with pytest.raises(ValidationError):
        InformationUnit(
            content="x",
            information_type=InformationType.FACT,
            agent_id=AgentRole.CRITIC,
            confidence=-0.1,
        )


def test_unit_empty_content_rejected():
    with pytest.raises(ValidationError):
        InformationUnit(
            content="   ",
            information_type=InformationType.FACT,
            agent_id=AgentRole.CRITIC,
        )


def test_set_embedding(unit):
    unit.set_embedding([0.1, -0.2, 0.3])
    assert unit.has_embedding
    assert unit.embedding_dim == 3
    assert unit.embedding == [0.1, -0.2, 0.3]  # stored as plain floats

    with pytest.raises(ValueError):
        unit.set_embedding([])


def test_unit_record_round_trip(unit):
    restored = InformationUnit.from_record(unit.to_record())
    assert restored == unit
    # serialised timestamps are strings; parsed back to datetime
    assert isinstance(unit.to_record()["timestamp"], str)
    assert restored.timestamp == unit.timestamp


def test_new_id_and_utc_now_formats():
    assert new_id("edge").startswith("edge_")
    now = utc_now()
    assert now.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# InformationEdge
# ---------------------------------------------------------------------------

def test_edge_happy_path(support_edge):
    assert support_edge.relationship_type is RelationshipType.SUPPORTS
    assert support_edge.is_support and not support_edge.is_contradiction
    assert support_edge.id.startswith("edge_")
    assert "SUPPORTS" in support_edge.direction_text()
    assert support_edge.evidence_or_rationale


def test_edge_self_loop_rejected(unit):
    with pytest.raises(ValidationError):
        InformationEdge(
            source_node_id=unit.id,
            target_node_id=unit.id,
            relationship_type=RelationshipType.SUPPORTS,
            confidence=0.9,
        )


def test_edge_unrelated_rejected():
    """UNRELATED is a classifier outcome, never a stored edge."""
    with pytest.raises(ValidationError):
        InformationEdge(
            source_node_id="iu_a",
            target_node_id="iu_b",
            relationship_type=RelationshipType.UNRELATED,
            confidence=0.95,
        )


def test_edge_confidence_bounds():
    with pytest.raises(ValidationError):
        InformationEdge(
            source_node_id="iu_a",
            target_node_id="iu_b",
            relationship_type=RelationshipType.CONTRADICTS,
            confidence=1.1,
        )


def test_edge_record_round_trip(support_edge):
    assert InformationEdge.from_record(support_edge.to_record()) == support_edge


def test_all_stored_edge_types_constructible():
    for rel in RelationshipType.stored_types():
        edge = InformationEdge(
            source_node_id="iu_a",
            target_node_id="iu_b",
            relationship_type=rel,
            confidence=0.7,
        )
        assert edge.relationship_type is rel


# ---------------------------------------------------------------------------
# orchestration types
# ---------------------------------------------------------------------------

def test_subtask_defaults():
    st = Subtask(description="Investigate supply data")
    assert st.status is SubtaskStatus.PENDING
    assert st.assigned_agent is AgentRole.RESEARCHER
    assert st.unit_ids == []
    assert st.id.startswith("st_")


def test_reasoning_context_prompt_block(unit, support_edge):
    from latent_workspace.models import ContextItem, ContextRelation, ReasoningContext

    item = ContextItem(
        unit=unit,
        relations=[
            ContextRelation(edge=support_edge, direction="outgoing"),
        ],
    )
    ctx = ReasoningContext(task_description="check supply claim", items=[item])

    block = ctx.as_prompt_block()
    assert unit.id in block
    assert "SUPPORTS(outgoing" in block
    assert ctx.unit_ids() == [unit.id]
    assert len(ctx) == 1

    empty = ReasoningContext(task_description="nothing")
    assert "no relevant workspace context" in empty.as_prompt_block()


def test_agent_result_success_flag():
    from latent_workspace.models import AgentResult

    ok = AgentResult(agent_id=AgentRole.CRITIC, units=[])
    assert ok.succeeded
    bad = AgentResult(agent_id=AgentRole.CRITIC, error="LLM timeout")
    assert not bad.succeeded


def test_orchestrator_result_defaults():
    from latent_workspace.models import OrchestratorResult

    res = OrchestratorResult(query="Why did prices rise?")
    assert res.final_answer == ""
    assert res.metrics.agent_calls == 0
    assert res.errors == []


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_settings_defaults_are_safe_for_offline_dev():
    s = Settings(_env_file=None)  # ignore any local .env
    assert s.mock_mode is True
    assert s.top_k == 8
    assert s.relation_threshold == 0.65
    assert 0.0 <= s.cosine_floor <= 1.0
    assert s.max_rounds == 4
    assert s.embedding_dim == 384
    assert s.vector_path.name == "vectors.faiss"
    assert s.db_path.suffix == ".sqlite3"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
