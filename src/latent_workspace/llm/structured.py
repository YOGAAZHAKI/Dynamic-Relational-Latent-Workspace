"""Structured-output helper: get valid JSON out of an LLM, with retries.

The problem it solves
---------------------
Language models often wrap JSON in markdown fences (```` ```json ````)
or add a sentence of prose around it. And sometimes they simply return
garbage. This module:

1. :func:`extract_json`     - digs the first JSON object/array out of any reply
2. :func:`complete_structured` - calls the LLM, parses, validates against a
   pydantic schema, and on failure sends the bad reply back asking for a
   correction (``llm_json_retries`` times) before raising
   :class:`LLMInvalidResponseError`.

Connection errors are NOT retried here — they propagate immediately so
the agent can report them (retrying a dead endpoint just wastes time).
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .base import ChatMessage, LLMClient, LLMInvalidResponseError, LLMResponse

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> object:
    """Return the first JSON structure found in ``text``.

    Accepts: raw JSON, fenced JSON, prose + JSON, leading commentary.
    Raises :class:`LLMInvalidResponseError` when nothing parses.
    """
    if not text or not text.strip():
        raise LLMInvalidResponseError("Model returned an empty response")

    cleaned = text.strip()
    fence = _FENCE_RE.search(cleaned)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(cleaned)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Fall back: scan for the first '{' or '[' and decode from there.
        for opener in ("{", "["):
            start = candidate.find(opener)
            if start == -1:
                continue
            try:
                obj, _end = json.JSONDecoder().raw_decode(candidate[start:])
                return obj
            except json.JSONDecodeError:
                continue

    preview = cleaned[:200]
    raise LLMInvalidResponseError(f"No valid JSON found in response: {preview!r}")


def complete_structured(
    llm: LLMClient,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    mock_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 2,
) -> tuple[T, list[LLMResponse]]:
    """Call the LLM until its reply validates as ``schema``.

    Returns ``(parsed, responses)`` — all responses so the agent can sum
    token usage across retries.

    On permanent failure raises :class:`LLMInvalidResponseError`.
    """
    chat: list[ChatMessage] = list(messages)
    responses: list[LLMResponse] = []
    last_error: Exception | None = None

    for _attempt in range(max_retries + 1):
        response = llm.complete(
            chat,
            mock_key=mock_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        responses.append(response)
        try:
            data = extract_json(response.text)
            parsed = schema.model_validate(data)
            return parsed, responses
        except (LLMInvalidResponseError, ValidationError) as exc:
            last_error = exc
            correction = (
                "Your previous reply could not be used. Error: "
                f"{exc}\n"
                "Respond with ONLY a single valid JSON object matching the "
                "required schema. No explanations, no markdown fences."
            )
            chat = [
                *chat,
                ChatMessage(role="assistant", content=response.text or "(empty)"),
                ChatMessage(role="user", content=correction),
            ]

    raise LLMInvalidResponseError(
        f"{schema.__name__}: could not parse a valid reply after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )
