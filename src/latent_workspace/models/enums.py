"""Shared enumerations: the vocabulary of the workspace.

These enums appear in every data model, agent output, graph edge and
evaluation metric, so they are defined once and imported everywhere.
"""

from __future__ import annotations

from enum import Enum


class RelationshipType(str, Enum):
    """Typed semantic relationship between two information units.

    Edge direction convention (used everywhere in this project):

        edge = (source, target, relationship_type)

    reads as ``source <relationship_type> target``:

        SUPPORTS     source supports target
        CONTRADICTS  source contradicts target
        REFINES      source refines / improves target
        DEPENDS_ON   source depends on target (source is valid only if target is)
        REDUNDANT    source says essentially the same thing as target
        UNRELATED    no meaningful relation (never stored as an edge)
    """

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    DEPENDS_ON = "DEPENDS_ON"
    REDUNDANT = "REDUNDANT"
    UNRELATED = "UNRELATED"

    @classmethod
    def stored_types(cls) -> tuple["RelationshipType", ...]:
        """Relationship types that are allowed to become graph edges.

        ``UNRELATED`` is a *prediction outcome* only: it tells the
        pipeline "skip this candidate". Storing it would fill the graph
        with meaningless edges.
        """
        return (
            cls.SUPPORTS,
            cls.CONTRADICTS,
            cls.REFINES,
            cls.DEPENDS_ON,
            cls.REDUNDANT,
        )


class InformationType(str, Enum):
    """What kind of claim an information unit carries."""

    HYPOTHESIS = "HYPOTHESIS"
    SUBTASK = "SUBTASK"
    EVIDENCE = "EVIDENCE"
    COUNTER_EVIDENCE = "COUNTER_EVIDENCE"
    REFINEMENT = "REFINEMENT"
    ASSUMPTION = "ASSUMPTION"
    FACT = "FACT"
    CONCLUSION = "CONCLUSION"
    QUESTION = "QUESTION"


class AgentRole(str, Enum):
    """Identifies which agent produced (or should handle) something."""

    PLANNER = "PLANNER"
    RESEARCHER = "RESEARCHER"
    CRITIC = "CRITIC"
    SYNTHESIZER = "SYNTHESIZER"


class SubtaskStatus(str, Enum):
    """Lifecycle of a subtask inside the orchestrator."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
