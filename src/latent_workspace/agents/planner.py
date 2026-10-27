"""Planner / Hypothesis agent.

Reads the user query, returns ``Subtask`` items for the orchestrator and
``HYPOTHESIS`` information units for the workspace graph.
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
    Subtask,
)
from .base import BaseAgent
from .prompts import PLANNER_SYSTEM, PLANNER_USER


class SubtaskSpec(BaseModel):
    description: str = Field(min_length=1)
    assigned_agent: AgentRole = AgentRole.RESEARCHER

    @field_validator("assigned_agent", mode="after")
    @classmethod
    def _only_worker_agents(cls, value: AgentRole) -> AgentRole:
        if value not in (AgentRole.RESEARCHER, AgentRole.CRITIC):
            raise ValueError(
                "assigned_agent must be RESEARCHER or CRITIC "
                "(the Planner and Synthesizer are not subtask workers)"
            )
        return value


class HypothesisSpec(BaseModel):
    content: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlanOutput(BaseModel):
    """JSON schema the Planner's reply must satisfy."""

    subtasks: list[SubtaskSpec] = Field(min_length=1, max_length=6)
    hypotheses: list[HypothesisSpec] = Field(min_length=1, max_length=6)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def system_prompt(self) -> str:
        return PLANNER_SYSTEM

    def _run(
        self,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
    ) -> AgentResult:
        # The Planner works from the raw query; context is accepted for
        # interface symmetry but not required on the first round.
        messages = self.build_messages(PLANNER_USER.replace("__QUERY__", task.strip()))
        plan, responses = complete_structured(
            self.llm,
            messages,
            PlanOutput,
            mock_key=self.mock_key,
            max_retries=self.settings.llm_json_retries,
        )

        subtasks = [
            Subtask(description=spec.description, assigned_agent=spec.assigned_agent)
            for spec in plan.subtasks
        ]
        units = [
            InformationUnit(
                content=hyp.content,
                information_type=InformationType.HYPOTHESIS,
                agent_id=self.role,
                confidence=hyp.confidence,
                metadata={"round": "planning"},
            )
            for hyp in plan.hypotheses
        ]
        return AgentResult(
            agent_id=self.role,
            subtask_id=subtask_id,
            units=units,
            subtasks=subtasks,
            raw_output=responses[-1].text,
            tokens_used=self.total_tokens(responses),
        )
