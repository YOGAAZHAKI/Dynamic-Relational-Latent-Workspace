"""Researcher / Evidence agent.

Reads a subtask + the retrieved reasoning context, emits ``EVIDENCE`` or
``FACT`` units carrying ``sources`` and ``depends_on`` metadata
(the raw material for future DEPENDS_ON edges).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..llm.structured import complete_structured
from ..models import (
    AgentResult,
    AgentRole,
    InformationType,
    InformationUnit,
    ReasoningContext,
)
from .base import BaseAgent
from .prompts import RESEARCHER_SYSTEM, RESEARCHER_USER

_ALLOWED_TYPES = {InformationType.EVIDENCE, InformationType.FACT}


class EvidenceSpec(BaseModel):
    content: str = Field(min_length=1)
    information_type: InformationType = InformationType.EVIDENCE
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("information_type", mode="after")
    @classmethod
    def _evidence_only(cls, value: InformationType) -> InformationType:
        if value not in _ALLOWED_TYPES:
            raise ValueError(
                "Researcher units must be EVIDENCE or FACT; "
                "hypotheses belong to the Planner"
            )
        return value


class ResearchOutput(BaseModel):
    evidence: list[EvidenceSpec] = Field(min_length=1, max_length=6)


class ResearcherAgent(BaseAgent):
    role = AgentRole.RESEARCHER

    def system_prompt(self) -> str:
        return RESEARCHER_SYSTEM

    def _run(
        self,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
    ) -> AgentResult:
        ctx = self.render_context(context, task)
        user = (
            RESEARCHER_USER
            .replace("__TASK__", task.strip())
            .replace("__CONTEXT__", ctx.as_prompt_block())
        )
        plan, responses = complete_structured(
            self.llm,
            self.build_messages(user),
            ResearchOutput,
            mock_key=self.mock_key,
            max_retries=self.settings.llm_json_retries,
        )

        units = [
            InformationUnit(
                content=spec.content,
                information_type=spec.information_type,
                agent_id=self.role,
                confidence=spec.confidence,
                metadata={
                    "sources": spec.sources,
                    "depends_on": spec.depends_on,
                    "subtask_id": subtask_id or "",
                },
            )
            for spec in plan.evidence
        ]
        return AgentResult(
            agent_id=self.role,
            subtask_id=subtask_id,
            units=units,
            raw_output=responses[-1].text,
            tokens_used=self.total_tokens(responses),
        )
