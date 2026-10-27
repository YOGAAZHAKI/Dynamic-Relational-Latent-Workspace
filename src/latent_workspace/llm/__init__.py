"""LLM client layer.

One tiny interface for every language-model call in the project::

    client = build_llm(settings)                 # mock or real, from .env
    reply  = client.complete(messages, mock_key="PLANNER")

- :class:`MockLLMClient`     -> fully offline, deterministic (tests, demo)
- :class:`OpenAICompatClient`-> any OpenAI-compatible API (OpenAI, OpenRouter,
                                Groq, Together, local Ollama/vLLM)
"""

from __future__ import annotations

from .base import (
    ChatMessage,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMResponse,
)
from .mock import MockLLMClient
from .structured import complete_structured, extract_json


def build_llm(settings=None):
    """Factory: returns the right client for the current settings.

    ``DLRW_MOCK_MODE=true``  -> MockLLMClient   (no key, no network)
    ``DLRW_MOCK_MODE=false`` -> OpenAICompatClient (reads .env)
    """
    from ..config import get_settings

    s = settings or get_settings()
    if s.mock_mode:
        return MockLLMClient()
    from .openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout,
    )


__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMResponse",
    "MockLLMClient",
    "OpenAICompatClient",
    "build_llm",
    "complete_structured",
    "extract_json",
]
