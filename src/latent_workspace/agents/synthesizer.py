"""Synthesizer agent: produces the final grounded answer.

Unlike the other agents it returns **plain text** (not JSON): its output
is the answer shown to the user, and it must cite workspace unit ids
like ``[iu_a1b2c3d4]``.
"""

from __future__ import annotations

from ..llm.base import LLMResponse
from ..models import AgentResult, AgentRole, ReasoningContext
from .base import BaseAgent
from .prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER


class SynthesizerAgent(BaseAgent):
    role = AgentRole.SYNTHESIZER

    def system_prompt(self) -> str:
        return SYNTHESIZER_SYSTEM

    def _run(
        self,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
    ) -> AgentResult:
        ctx = self.render_context(context, task)
        user = (
            SYNTHESIZER_USER
            .replace("__QUERY__", task.strip())
            .replace("__CONTEXT__", ctx.as_prompt_block())
        )
        response: LLMResponse = self.llm.complete(
            self.build_messages(user),
            mock_key=self.mock_key,
        )
        answer = (response.text or "").strip()
        if not answer:
            from ..llm.base import LLMInvalidResponseError

            raise LLMInvalidResponseError("Synthesizer returned an empty answer")
        return AgentResult(
            agent_id=self.role,
            subtask_id=subtask_id,
            answer=answer,
            raw_output=response.text,
            tokens_used=response.total_tokens,
        )
