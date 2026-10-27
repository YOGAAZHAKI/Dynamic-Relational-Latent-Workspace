"""Orchestration data types.

These describe the *conversation between components* rather than the
knowledge itself:

- ``Subtask``         what the Planner asked for
- ``AgentResult``     what one agent returned
- ``ContextItem`` /
  ``ReasoningContext`` what the graph retriever hands to the next agent
- ``RunMetrics`` /
  ``OrchestratorResult`` the final report of one full run
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .edge import InformationEdge
from .enums import AgentRole, RelationshipType, SubtaskStatus
from .information_unit import InformationUnit, new_id


class Subtask(BaseModel):
    """One work item produced by the Planner agent."""

    id: str = Field(default_factory=lambda: new_id("st"))
    description: str = Field(min_length=1)
    assigned_agent: AgentRole = AgentRole.RESEARCHER
    status: SubtaskStatus = SubtaskStatus.PENDING
    unit_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class AgentResult(BaseModel):
    """Structured return value of a single agent invocation."""

    agent_id: AgentRole
    subtask_id: str | None = None
    units: list[InformationUnit] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    answer: str = ""
    raw_output: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ContextRelation(BaseModel):
    """One edge annotated with how it relates to the retrieved node."""

    edge: InformationEdge
    direction: Literal["incoming", "outgoing"]


class ProvenanceRef(BaseModel):
    """One hop of a unit's grounding chain, condensed for prompts.

    This is *predicted* provenance (typed edges with confidence), not
    verified factual lineage.
    """

    unit_id: str
    content_preview: str
    relationship_type: RelationshipType
    confidence: float
    depth: int


class ContextItem(BaseModel):
    """A workspace unit plus the edges that connect it to the subgraph."""

    unit: InformationUnit
    relations: list[ContextRelation] = Field(default_factory=list)
    #: how the unit relates to the query's anchor unit (filled by the
    #: SubgraphRetriever; "neutral" when there is no anchor relation)
    stance: Literal["supporting", "contradicting", "neutral"] = "neutral"
    #: short grounding chain for this unit, when the retriever could find one
    provenance: list[ProvenanceRef] = Field(default_factory=list)

    @property
    def relation_labels(self) -> list[str]:
        """``['CONTRADICTS(in, 0.81)', ...]`` — used in agent prompts so
        the agent sees *typed, confidence-labeled* links, never bare text.
        """
        labels = []
        for rel in self.relations:
            labels.append(
                f"{rel.edge.relationship_type.value}"
                f"({rel.direction}, conf={rel.edge.confidence:.2f})"
            )
        return labels


class ReasoningContext(BaseModel):
    """The relevant subgraph slice handed to the next agent.

    ``items`` are ordered by retrieval score (best first). ``truncated``
    flags that the token budget was hit — agents are told not to treat
    the list as exhaustive.

    Chunk 6 additions (all defaulted, so older constructors still work):

    - ``anchor_id``: the highest-scoring *seed* the query matched; the
      "center" of the subgraph that stances are measured against.
    - ``supporting_ids`` / ``contradicting_ids``: items measured against
      the anchor. Conflicts are *never* silently dropped — if the token
      budget cut them, they land in ``excluded_unit_ids`` too and
      ``truncated`` is set.
    - ``excluded_unit_ids``: reached by retrieval but not shown (budget /
      cap). Downstream code can re-fetch them with a bigger budget.
    """

    task_description: str
    items: list[ContextItem] = Field(default_factory=list)
    truncated: bool = False
    token_estimate: int = 0
    anchor_id: str | None = None
    supporting_ids: list[str] = Field(default_factory=list)
    contradicting_ids: list[str] = Field(default_factory=list)
    excluded_unit_ids: list[str] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def unit_ids(self) -> list[str]:
        return [item.unit.id for item in self.items]

    def has_conflicts(self) -> bool:
        """True when the retrieved slice contains contradicting evidence."""
        return bool(self.contradicting_ids)

    def supporting_items(self) -> list[ContextItem]:
        return [i for i in self.items if i.stance == "supporting"]

    def contradicting_items(self) -> list[ContextItem]:
        return [i for i in self.items if i.stance == "contradicting"]

    def as_prompt_block(self) -> str:
        """Render the context as text for inclusion in an agent prompt."""
        if not self.items:
            return "(no relevant workspace context was retrieved)"
        lines: list[str] = []
        for i, item in enumerate(self.items, start=1):
            labels = ", ".join(item.relation_labels) or "no edges"
            stance = (
                f" [{item.stance.upper()}]" if item.stance != "neutral" else ""
            )
            lines.append(
                f"{i}. [{item.unit.id}] ({item.unit.information_type.value}, "
                f"conf={item.unit.confidence:.2f}){stance} {item.unit.content}"
                f"\n   relations: {labels}"
            )
            if item.provenance:
                chain = " <- ".join(
                    f"{p.unit_id} {p.relationship_type.value} "
                    f"(conf={p.confidence:.2f})"
                    for p in item.provenance
                )
                lines.append(f"   provenance: {chain}")
        note = "\n(note: context was truncated)" if self.truncated else ""
        return "\n".join(lines) + note


class RunMetrics(BaseModel):
    """Measured quantities for one orchestrator run."""

    latency_ms: float = 0.0
    total_tokens: int = 0
    agent_calls: int = 0
    rounds: int = 0
    nodes_added: int = 0
    edges_added: int = 0


class OrchestratorResult(BaseModel):
    """Everything one end-to-end run produced (feeds the Streamlit UI
    and the evaluation harness in later chunks)."""

    query: str
    final_answer: str = ""
    subtasks: list[Subtask] = Field(default_factory=list)
    units: list[InformationUnit] = Field(default_factory=list)
    edges: list[InformationEdge] = Field(default_factory=list)
    contexts: list[ReasoningContext] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    errors: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
