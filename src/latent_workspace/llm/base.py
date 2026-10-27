"""The LLM client interface shared by mock and real implementations.

Beginner map
------------
ChatMessage   one chat turn (role = system/user/assistant)
LLMResponse   what a call returns: text + token counts + latency
LLMClient     the Protocol (= interface) every client satisfies
LLMError      base class for every LLM problem; agents catch this
  LLMConnectionError      network / auth / provider trouble
  LLMInvalidResponseError the model said something we cannot parse
"""

from __future__ import annotations

from typing import Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One message in the conversation sent to the model."""

    role: ChatRole
    content: str = Field(min_length=1)


class LLMResponse(BaseModel):
    """Normalized result of a single ``complete()`` call."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class LLMError(Exception):
    """Base class for all LLM client failures."""


class LLMConnectionError(LLMError):
    """The call never produced usable output (network, auth, provider)."""


class LLMInvalidResponseError(LLMError):
    """The model replied, but the reply cannot be parsed/validated."""


@runtime_checkable
class LLMClient(Protocol):
    """Interface every LLM client implements.

    ``mock_key`` is a hint used *only* by the mock client to pick a
    canned reply; real clients ignore it entirely.
    """

    model_name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        mock_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
