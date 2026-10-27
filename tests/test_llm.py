"""Tests for the LLM client layer (Chunk 2).

All offline: no API keys, no network.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from latent_workspace.config import Settings
from latent_workspace.llm import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMResponse,
    MockLLMClient,
    build_llm,
)
from latent_workspace.llm.openai_compat import OpenAICompatClient

MESSAGES = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="Task:\nInvestigate supply data"),
]


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_build_llm_returns_mock_when_mock_mode():
    s = Settings(_env_file=None, mock_mode=True)
    client = build_llm(s)
    assert isinstance(client, MockLLMClient)


def test_build_llm_returns_real_client_when_mock_mode_off():
    s = Settings(
        _env_file=None,
        mock_mode=False,
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
    )
    client = build_llm(s)
    assert isinstance(client, OpenAICompatClient)
    assert client.model_name == "gpt-4o-mini"


def test_build_llm_real_client_requires_model_name():
    s = Settings(_env_file=None, mock_mode=False, llm_model="")
    with pytest.raises(LLMError):
        build_llm(s)


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

def test_mock_client_is_deterministic():
    a = MockLLMClient().complete(MESSAGES, mock_key="PLANNER")
    b = MockLLMClient().complete(MESSAGES, mock_key="PLANNER")
    assert a.text == b.text          # same input -> byte-identical output
    assert a.model == "mock-llm"
    assert a.total_tokens > 0


def test_mock_client_respects_scripted_replies():
    client = MockLLMClient(scripted={"RESEARCHER": "custom reply"})
    out = client.complete(MESSAGES, mock_key="RESEARCHER")
    assert out.text == "custom reply"


def test_mock_client_records_calls_for_inspection():
    client = MockLLMClient()
    client.complete(MESSAGES, mock_key="CRITIC")
    client.complete(MESSAGES, mock_key="CRITIC")
    assert len(client.calls) == 2
    assert client.calls[0]["mock_key"] == "CRITIC"
    assert "Investigate supply data" in client.calls[0]["user_text"]


def test_mock_client_can_simulate_outage():
    client = MockLLMClient(fail_keys=["PLANNER"])
    with pytest.raises(LLMConnectionError):
        client.complete(MESSAGES, mock_key="PLANNER")
    # keys not in fail_keys still work
    assert client.complete(MESSAGES, mock_key="RESEARCHER").text


def test_mock_client_unknown_key_still_returns_text():
    out = MockLLMClient().complete(MESSAGES, mock_key="SOMETHING_ELSE")
    assert out.text  # non-empty, never crashes


def test_mock_client_replies_are_valid_json_for_json_agents():
    import json

    for key in ("PLANNER", "RESEARCHER", "CRITIC"):
        out = MockLLMClient().complete(MESSAGES, mock_key=key)
        json.loads(out.text)  # must not raise


# ---------------------------------------------------------------------------
# OpenAICompatClient (with an injected stub -> no network)
# ---------------------------------------------------------------------------

class _StubChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class _StubUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class _StubResponse:
    def __init__(self, text):
        self.choices = [_StubChoice(text)] if text is not None else []
        self.usage = _StubUsage()


class _StubCompletions:
    def __init__(self, text="hi", error=None):
        self._text = text
        self._error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return _StubResponse(self._text)


def _stub_client(text="hi", error=None):
    stub = _StubCompletions(text=text, error=error)
    root = type("Root", (), {"chat": type("C", (), {"completions": stub})()})()
    return OpenAICompatClient(
        base_url="http://x", api_key="k", model="stub-model", client=root
    ), stub


def test_openai_client_maps_response():
    client, stub = _stub_client(text="hello there")
    out = client.complete(MESSAGES, mock_key="IGNORED_BY_REAL")
    assert out.text == "hello there"
    assert out.total_tokens == 30 and out.prompt_tokens == 10
    assert out.model == "stub-model"
    assert out.latency_ms >= 0
    # our messages were forwarded correctly
    assert stub.kwargs["model"] == "stub-model"
    assert stub.kwargs["messages"][0]["role"] == "system"


def test_openai_client_wraps_connection_errors():
    client, _ = _stub_client(error=RuntimeError("boom"))
    with pytest.raises(LLMConnectionError) as exc:
        client.complete(MESSAGES)
    assert "boom" in str(exc.value)


def test_openai_client_empty_choices_is_invalid_response():
    from latent_workspace.llm.base import LLMInvalidResponseError

    client, _ = _stub_client(text=None)
    with pytest.raises(LLMInvalidResponseError):
        client.complete(MESSAGES)


def test_openai_client_requires_model():
    with pytest.raises(LLMError):
        OpenAICompatClient(base_url="http://x", api_key="k", model="")


def test_llm_response_defaults():
    r = LLMResponse(text="x")
    assert r.total_tokens == 0 and r.model == ""
