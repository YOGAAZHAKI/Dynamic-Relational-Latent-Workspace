"""Shared pytest fixtures.

Everything here is offline and deterministic — no API keys, no model
downloads. Later chunks add mock LLM / embedder fixtures alongside these.
"""

from __future__ import annotations

import pytest

from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
    RelationshipType,
)


@pytest.fixture
def unit() -> InformationUnit:
    """A typical evidence unit."""
    return InformationUnit(
        content="Global chip supply fell 12% in Q3 due to fab shutdowns.",
        information_type=InformationType.EVIDENCE,
        agent_id=AgentRole.RESEARCHER,
        confidence=0.8,
        metadata={"subtask_id": "st_0001"},
    )


@pytest.fixture
def other_unit() -> InformationUnit:
    """A hypothesis that the evidence above could support."""
    return InformationUnit(
        content="The price increase was driven by a supply shortage.",
        information_type=InformationType.HYPOTHESIS,
        agent_id=AgentRole.PLANNER,
        confidence=0.6,
    )


@pytest.fixture
def support_edge(unit, other_unit):
    """edge: evidence SUPPORTS hypothesis (source -> target reads
    'source supports target')."""
    from latent_workspace.models import InformationEdge

    return InformationEdge(
        source_node_id=unit.id,
        target_node_id=other_unit.id,
        relationship_type=RelationshipType.SUPPORTS,
        confidence=0.82,
        evidence_or_rationale="A 12% supply fall is direct support for the "
        "supply-shortage hypothesis.",
    )
