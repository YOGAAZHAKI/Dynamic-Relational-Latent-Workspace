"""Client for any OpenAI-compatible chat completions endpoint.

Works with: OpenAI, OpenRouter, Groq, Together, Mistral, and local
servers (Ollama, vLLM, LM Studio) — anything that serves
``POST {base_url}/chat/completions``.

Ollama example (.env)::

    DLRW_MOCK_MODE=false
    DLRW_LLM_BASE_URL=http://localhost:11434/v1
    DLRW_LLM_API_KEY=ollama          # required by the SDK, value ignored
    DLRW_LLM_MODEL=llama3.1
"""

from __future__ import annotations

import time
from typing import Sequence

from .base import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMResponse,
)

try:  # optional at import time so the mock path never needs it
    import openai as _openai
except ImportError:  # pragma: no cover
    _openai = None


class OpenAICompatClient:
    """Thin wrapper around ``openai.OpenAI`` with normalized outputs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        client=None,  # injection point for tests (no network needed)
    ) -> None:
        if not model:
            raise LLMError("llm_model must not be empty (set DLRW_LLM_MODEL)")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
            return
        if _openai is None:
            raise LLMError(
                "The 'openai' package is required for real LLM mode. "
                "Install it:  python -m pip install openai"
            )
        # Local servers (Ollama...) never check the key, but the SDK wants
        # a non-empty string, so fall back to a harmless placeholder.
        self._client = _openai.OpenAI(
            base_url=base_url or None,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=2,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        mock_key: str | None = None,  # ignored: real API
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
        except Exception as exc:  # provider-specific error classes
            name = type(exc).__name__
            hint = ""
            if "Authentication" in name or "Permission" in name:
                hint = " Check DLRW_LLM_API_KEY in your .env file."
            elif "Connection" in name or "Timeout" in name or "APIConnection" in name:
                hint = " Check DLRW_LLM_BASE_URL and your network."
            raise LLMConnectionError(
                f"LLM call to '{self.model_name}' failed ({name}): {exc}.{hint}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not getattr(response, "choices", None):
            raise LLMInvalidResponseError(
                f"Provider returned no completion choices (model={self.model_name})"
            )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            latency_ms=latency_ms,
        )
    