"""Agent base class.

Design: **template method**.

- ``run()`` (here)  -> timing, token totals, crash-proof error handling.
- ``_run()`` (subclass) -> prompt building + LLM call + parsing.

Because ``run()`` catches everything, one agent failing never kills a
pipeline run — the orchestrator sees ``AgentResult.error`` and moves on.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..config import Settings, get_settings
from ..llm.base import ChatMessage, LLMClient, LLMError
from ..models import AgentResult, AgentRole, ReasoningContext


class BaseAgent(ABC):
    """Common machinery for every agent."""

    role: AgentRole

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    # -- public entry point ---------------------------------------------
    def run(
        self,
        task: str,
        context: ReasoningContext | None = None,
        *,
        subtask_id: str | None = None,
    ) -> AgentResult:
        """Execute the agent safely: always returns an AgentResult."""
        started = time.perf_counter()
        try:
            result = self._run(task, context, subtask_id)
        except LLMError as exc:
            result = AgentResult(
                agent_id=self.role,
                subtask_id=subtask_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # unexpected bug: still do not crash
            result = AgentResult(
                agent_id=self.role,
                subtask_id=subtask_id,
                error=f"unexpected {type(exc).__name__}: {exc}",
            )
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    @abstractmethod
    def _run(
        self,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
    ) -> AgentResult:
        """Agent-specific logic (implemented by each subclass)."""

    # -- shared helpers ---------------------------------------------------
    @abstractmethod
    def system_prompt(self) -> str: ...

    def build_messages(self, user_content: str) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.system_prompt()),
            ChatMessage(role="user", content=user_content),
        ]

    @staticmethod
    def render_context(context: ReasoningContext | None, task: str) -> ReasoningContext:
        """Treat a missing context as an empty one (not an error)."""
        if context is not None:
            return context
        return ReasoningContext(task_description=task)

    @staticmethod
    def total_tokens(responses) -> int:
        return sum(r.total_tokens for r in responses)

    @property
    def mock_key(self) -> str:
        """Hint passed to the mock LLM so it knows which canned reply to use."""
        return self.role.value
