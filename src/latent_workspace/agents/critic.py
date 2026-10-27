"""Critic / Verification agent.

Scrutinizes hypotheses + evidence from the reasoning context and emits
``COUNTER_EVIDENCE`` (or ``REFINEMENT``/``CONCLUSION``) units. Each unit
carries a ``target`` in metadata naming what the critique acts on —
the relation-detector will later propose CONTRADICTS/REFINES edges
toward it.
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
from .prompts import CRITIC_SYSTEM, CRITIC_USER

_CRITIC_TYPES = {
    InformationType.COUNTER_EVIDENCE,
    InformationType.REFINEMENT,
    InformationType.CONCLUSION,
    InformationType.FACT,
}


class CritiqueSpec(BaseModel):
    content: str = Field(min_length=1)
    information_type: InformationType = InformationType.COUNTER_EVIDENCE
    target: str | None = Field(
        default=None,
        description="What this critique acts on: a claim phrase or unit id.",
    )
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)

    @field_validator("information_type", mode="after")
    @classmethod
    def _critic_types_only(cls, value: InformationType) -> InformationType:
        if value not in _CRITIC_TYPES:
            raise ValueError(
                "Critic units must be COUNTER_EVIDENCE, REFINEMENT, "
                "CONCLUSION or FACT"
            )
        return value


class CritiqueOutput(BaseModel):
    critiques: list[CritiqueSpec] = Field(min_length=1, max_length=6)


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC

    def system_prompt(self) -> str:
        return CRITIC_SYSTEM

    def _run(
        self,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
    ) -> AgentResult:
        ctx = self.render_context(context, task)
        user = (
            CRITIC_USER
            .replace("__TASK__", task.strip())
            .replace("__CONTEXT__", ctx.as_prompt_block())
        )
        plan, responses = complete_structured(
            self.llm,
            self.build_messages(user),
            CritiqueOutput,
            mock_key=self.mock_key,
            max_retries=self.settings.llm_json_retries,
        )

        units = []
        for spec in plan.critiques:
            metadata: dict = {"subtask_id": subtask_id or "", "issues": spec.issues}
            if spec.target:
                metadata["target"] = spec.target
            units.append(
                InformationUnit(
                    content=spec.content,
                    information_type=spec.information_type,
                    agent_id=self.role,
                    confidence=spec.confidence,
                    metadata=metadata,
                )
            )
        return AgentResult(
            agent_id=self.role,
            subtask_id=subtask_id,
            units=units,
            raw_output=responses[-1].text,
            tokens_used=self.total_tokens(responses),
        )
