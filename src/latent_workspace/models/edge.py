"""InformationEdge: a typed, confidence-scored relationship between units.

Direction convention (project-wide, also documented in
``RelationshipType``):

    edge (source --relationship_type--> target)

reads as ``source <relationship_type> target``:

    ("supply shock",  SUPPORTS,    "prices rose")   -> supply shock supports prices-rose
    ("no shock in Q2", CONTRADICTS, "prices rose")  -> contradicts it
    ("prices rose 4%", REFINES,    "prices rose")   -> sharper version of it
    ("claim X", DEPENDS_ON, "statistic Y")          -> claim X needs Y to hold

So a *newer* unit (e.g. a critic's counter-evidence) points *at* the
older unit it acts on.

``UNRELATED`` is deliberately NOT storable: it is a classifier outcome
meaning "create no edge".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from .enums import RelationshipType
from .information_unit import new_id, utc_now


class InformationEdge(BaseModel):
    """A relationship edge in the relational workspace graph."""

    id: str = Field(default_factory=lambda: new_id("edge"))
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_or_rationale: str = Field(
        default="",
        description="Why the classifier believed this relation (from the "
        "relation detector or the agent).",
    )
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_invariants(self) -> "InformationEdge":
        if self.source_node_id == self.target_node_id:
            raise ValueError("Self-edges are not allowed")
        if self.relationship_type is RelationshipType.UNRELATED:
            raise ValueError(
                "UNRELATED edges must never be stored; skip the candidate instead"
            )
        return self

    @property
    def is_contradiction(self) -> bool:
        return self.relationship_type is RelationshipType.CONTRADICTS

    @property
    def is_support(self) -> bool:
        return self.relationship_type is RelationshipType.SUPPORTS

    def direction_text(self) -> str:
        """Human-readable statement of the direction convention."""
        return (
            f"{self.source_node_id} --{self.relationship_type.value}--> "
            f"{self.target_node_id} (conf={self.confidence:.2f})"
        )

    def to_record(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_record(cls, data: dict) -> "InformationEdge":
        return cls.model_validate(data)
