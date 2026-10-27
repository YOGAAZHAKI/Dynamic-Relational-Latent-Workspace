"""Relation detection: the typed-relationship classifier.

Why this layer exists
---------------------
Embedding similarity says two statements are *about the same topic*.
It can never tell whether one SUPPORTS or CONTRADICTS the other
("vaccines prevent disease" vs "vaccines cause disease" are near
neighbors). This package *reads* statement pairs and predicts a typed
relationship with a confidence and a human-readable rationale.

Pipeline position::

    candidate pairs (FAISS, Chunk 3)
        -> RelationDetector.detect(source, target)   <-- this chunk
        -> threshold + UNRELATED filter
        -> InformationEdge added to the graph (Chunk 5)

Edge direction convention (identical to :mod:`latent_workspace.models.edge`)
-------------------------------------------------------------------------
``detect(source_text, target_text)`` answers::

    "How does SOURCE relate to TARGET?"

and :meth:`RelationPrediction.to_edge` emits::

    source_node_id --RELATIONSHIP--> target_node_id

so a newer unit (e.g. critic counter-evidence) points *at* the older unit
it acts on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..models import InformationEdge, InformationUnit, RelationshipType


class RelationDetectorError(Exception):
    """Classifier failed on a pair (malformed reply, bad type, outage...)."""


class RelationPrediction(BaseModel):
    """Classifier output for one (source, target) pair."""

    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        default="",
        description="Why the classifier chose this type (shown in UI/edge).",
    )
    detector: str = Field(
        default="unknown",
        description="Which detector produced this (mock / llm:model / nli).",
    )

    @property
    def is_storable(self) -> bool:
        """UNRELATED is a prediction outcome, never a stored edge."""
        return self.relationship_type is not RelationshipType.UNRELATED

    def accepts(self, threshold: float) -> bool:
        """Gate used by the workspace: typed relation above threshold."""
        return self.is_storable and self.confidence >= threshold

    def to_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        threshold: float = 0.65,
    ) -> InformationEdge | None:
        """Convert to a stored edge, or ``None`` if filtered out.

        Filters: ``UNRELATED`` predictions, and any prediction whose
        confidence is below ``threshold``. Similarity never enters here —
        only the classifier's judgment does.
        """
        if not self.accepts(threshold):
            return None
        rationale = f"[{self.detector}] {self.rationale}".strip()
        return InformationEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=self.relationship_type,
            confidence=self.confidence,
            evidence_or_rationale=rationale,
        )

    def as_label(self) -> str:
        return (
            f"{self.relationship_type.value} "
            f"(conf={self.confidence:.2f}, detector={self.detector})"
        )


class RelationDetector(ABC):
    """Interface every detector implements."""

    name: Literal["mock", "llm", "nli", "custom"] = "custom"

    @abstractmethod
    def detect(self, source_text: str, target_text: str) -> RelationPrediction:
        """Classify how SOURCE relates to TARGET (see module docstring)."""

    def detect_units(
        self,
        source: InformationUnit,
        target: InformationUnit,
    ) -> RelationPrediction:
        """Convenience wrapper around :meth:`detect` using unit contents."""
        return self.detect(source.content, target.content)
