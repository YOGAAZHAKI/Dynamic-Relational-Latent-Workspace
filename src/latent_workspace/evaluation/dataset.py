"""Small, deterministic evaluation dataset (chunk 9).

Each case defines a mini-workspace and the things a good system should
surface. Markers are plain substrings rather than model judgements, so
scoring is reproducible and cheap:

- ``relevant_markers``  a unit containing any of these is *relevant*
- ``conflict_markers``  a relevant unit containing any of these is
  *counter-evidence* that a good answer must not silently drop
- ``key_points``        short phrases an answer should cover (or cite a
  unit containing)

The cases are deliberately small: the harness measures *retrieval and
grounding behaviour*, not broad world knowledge.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..models import AgentRole, InformationType


class EvalUnit(BaseModel):
    """One pre-existing unit in a case workspace."""

    content: str = Field(min_length=1)
    information_type: InformationType = InformationType.EVIDENCE
    agent_id: AgentRole = AgentRole.RESEARCHER
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class EvalCase(BaseModel):
    """A query plus the workspace and expectations used to score it."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    units: list[EvalUnit] = Field(min_length=1)
    relevant_markers: list[str] = Field(min_length=1)
    conflict_markers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(min_length=1)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("case id must be alphanumeric/underscore/dash")
        return value

    # -- derived helpers --------------------------------------------------
    def is_relevant(self, content: str) -> bool:
        text = (content or "").lower()
        return any(marker.lower() in text for marker in self.relevant_markers)

    def is_conflict(self, content: str) -> bool:
        text = (content or "").lower()
        return any(marker.lower() in text for marker in self.conflict_markers)

    @property
    def relevant_units(self) -> list[EvalUnit]:
        return [u for u in self.units if self.is_relevant(u.content)]

    @property
    def conflict_units(self) -> list[EvalUnit]:
        return [u for u in self.units if self.is_conflict(u.content)]


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="chip_prices",
        query="Why did global chip prices rise in 2025?",
        description="Supply shortage with one dissenting report.",
        units=[
            EvalUnit(
                content=(
                    "Global chip supply fell 12% in Q3 2025 after several "
                    "fabrication plants shut down."
                ),
                information_type=InformationType.EVIDENCE,
                confidence=0.82,
            ),
            EvalUnit(
                content="The chip price increase was driven by a supply shortage.",
                information_type=InformationType.HYPOTHESIS,
                agent_id=AgentRole.PLANNER,
                confidence=0.6,
            ),
            EvalUnit(
                content="Some analysts report that chip prices did not rise in Q3 2025.",
                information_type=InformationType.COUNTER_EVIDENCE,
                agent_id=AgentRole.CRITIC,
                confidence=0.7,
            ),
            EvalUnit(
                content="Prices for advanced chips rose approximately 18% year over year.",
                information_type=InformationType.REFINEMENT,
                agent_id=AgentRole.CRITIC,
                confidence=0.76,
            ),
            EvalUnit(
                content="The local football team won its championship final.",
                information_type=InformationType.FACT,
                confidence=0.9,
            ),
        ],
        relevant_markers=["chip", "prices", "supply"],
        conflict_markers=["did not rise"],
        key_points=["supply fell", "prices rose", "supply shortage"],
    ),
    EvalCase(
        id="vaccines",
        query="Do vaccines prevent infection?",
        description="Trial evidence plus a contradictory trial report.",
        units=[
            EvalUnit(
                content=(
                    "Randomized trials confirm that vaccines prevent infection "
                    "in most recipients."
                ),
                confidence=0.85,
            ),
            EvalUnit(
                content="Vaccines prevent infection.",
                information_type=InformationType.HYPOTHESIS,
                agent_id=AgentRole.PLANNER,
                confidence=0.6,
            ),
            EvalUnit(
                content=(
                    "One study found that vaccines do not prevent infection "
                    "in the exposed trial group."
                ),
                information_type=InformationType.COUNTER_EVIDENCE,
                agent_id=AgentRole.CRITIC,
                confidence=0.72,
            ),
            EvalUnit(
                content="Vaccine effectiveness was 87% against symptomatic infection.",
                information_type=InformationType.REFINEMENT,
                agent_id=AgentRole.CRITIC,
                confidence=0.78,
            ),
            EvalUnit(
                content="The city opened a new library on Monday.",
                information_type=InformationType.FACT,
                confidence=0.9,
            ),
        ],
        relevant_markers=["vaccine", "infection"],
        conflict_markers=["do not prevent"],
        key_points=["prevent infection", "vaccine effectiveness", "trials"],
    ),
    EvalCase(
        id="remote_work",
        query="Does remote work reduce team productivity?",
        description="Mixed productivity findings that require conflict surfacing.",
        units=[
            EvalUnit(
                content=(
                    "A 2024 survey of 2,000 teams found remote work reduced "
                    "productivity by 8%."
                ),
                confidence=0.8,
            ),
            EvalUnit(
                content="Remote work reduces team productivity.",
                information_type=InformationType.HYPOTHESIS,
                agent_id=AgentRole.PLANNER,
                confidence=0.55,
            ),
            EvalUnit(
                content=(
                    "Other studies show remote work did not reduce team "
                    "productivity."
                ),
                information_type=InformationType.COUNTER_EVIDENCE,
                agent_id=AgentRole.CRITIC,
                confidence=0.74,
            ),
            EvalUnit(
                content=(
                    "Productivity effects of remote work depend on team size "
                    "and management quality."
                ),
                information_type=InformationType.REFINEMENT,
                agent_id=AgentRole.CRITIC,
                confidence=0.69,
            ),
            EvalUnit(
                content="The office cafeteria serves lunch until 2pm.",
                information_type=InformationType.FACT,
                confidence=0.9,
            ),
        ],
        relevant_markers=["remote", "productivity"],
        conflict_markers=["did not reduce"],
        key_points=["productivity", "remote work", "survey"],
    ),
    EvalCase(
        id="ev_adoption",
        query="What drove electric vehicle adoption in 2025?",
        description="Battery-cost explanation with a counter-claim.",
        units=[
            EvalUnit(
                content=(
                    "Falling battery prices cut electric vehicle costs by 15% "
                    "in 2025."
                ),
                confidence=0.83,
            ),
            EvalUnit(
                content=(
                    "Electric vehicle adoption in 2025 was driven by lower "
                    "battery costs."
                ),
                information_type=InformationType.HYPOTHESIS,
                agent_id=AgentRole.PLANNER,
                confidence=0.6,
            ),
            EvalUnit(
                content="Analysts note that battery prices did not fall in 2025.",
                information_type=InformationType.COUNTER_EVIDENCE,
                agent_id=AgentRole.CRITIC,
                confidence=0.7,
            ),
            EvalUnit(
                content=(
                    "Electric vehicle adoption rose 22% in 2025 across major "
                    "markets."
                ),
                information_type=InformationType.EVIDENCE,
                confidence=0.8,
            ),
            EvalUnit(
                content="A new stadium was completed in the city.",
                information_type=InformationType.FACT,
                confidence=0.9,
            ),
        ],
        relevant_markers=["electric", "vehicle", "battery", "adoption"],
        conflict_markers=["did not fall"],
        key_points=["battery", "adoption rose", "electric vehicle"],
    ),
)
