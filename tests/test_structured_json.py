"""Tests for JSON extraction + structured retry (Chunk 2)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from latent_workspace.llm import (
    ChatMessage,
    LLMConnectionError,
    LLMInvalidResponseError,
    MockLLMClient,
    complete_structured,
    extract_json,
)

MSGS = [ChatMessage(role="user", content="Task:\nbuild it")]


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_plain_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_markdown_fenced():
    text = 'Sure!\n```json\n{"evidence": [{"content": "x"}]}\n```\nHope that helps.'
    assert extract_json(text) == {"evidence": [{"content": "x"}]}


def test_extract_json_prose_around_object():
    text = 'Here is your plan: {"subtasks": []} — let me know!'
    assert extract_json(text) == {"subtasks": []}


def test_extract_json_nested_with_trailing_comment():
    text = '{"a": {"b": [1, 2]}}  # done'
    assert extract_json(text) == {"a": {"b": [1, 2]}}


def test_extract_json_empty_response_raises():
    with pytest.raises(LLMInvalidResponseError):
        extract_json("")


def test_extract_json_no_json_raises():
    with pytest.raises(LLMInvalidResponseError):
        extract_json("I am sorry, I cannot help with that.")


# ---------------------------------------------------------------------------
# complete_structured: schema + retries
# ---------------------------------------------------------------------------

class _Spec(BaseModel):
    content: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class _Output(BaseModel):
    evidence: list[_Spec] = Field(min_length=1)


class _StubLLM:
    """Returns a scripted sequence of replies, then repeats the last one."""

    model_name = "stub"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, *, mock_key=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else "STUB_EXHAUSTED"
        from latent_workspace.llm import LLMResponse

        return LLMResponse(text=reply, model=self.model_name, total_tokens=5)


VALID = '{"evidence": [{"content": "unit one", "confidence": 0.7}]}'


def test_complete_structured_success_first_try():
    llm = _StubLLM([VALID])
    parsed, responses = complete_structured(llm, MSGS, _Output, max_retries=2)
    assert parsed.evidence[0].content == "unit one"
    assert len(responses) == 1
    assert len(llm.calls) == 1


def test_complete_structured_retries_then_succeeds():
    llm = _StubLLM(["this is not json at all", VALID])
    parsed, responses = complete_structured(llm, MSGS, _Output, max_retries=2)
    assert len(responses) == 2
    # second call included assistant + corrective user messages
    assert len(llm.calls[1]) == len(MSGS) + 2
    assert llm.calls[1][-1].role == "user"
    assert "ONLY a single valid JSON object" in llm.calls[1][-1].content


def test_complete_structured_gives_up_after_retries():
    llm = _StubLLM(["garbage", "more garbage", "even more garbage", "stale"])
    with pytest.raises(LLMInvalidResponseError) as exc:
        complete_structured(llm, MSGS, _Output, max_retries=2)
    assert "3 attempt" in str(exc.value)
    assert len(llm.calls) == 3


def test_complete_structured_validates_schema_fields():
    # parses as JSON but violates schema (empty evidence list)
    llm = _StubLLM(['{"evidence": []}'])
    with pytest.raises(LLMInvalidResponseError):
        complete_structured(llm, MSGS, _Output, max_retries=0)


def test_complete_structured_rejects_out_of_range_confidence():
    bad = '{"evidence": [{"content": "x", "confidence": 9}]}'
    llm = _StubLLM([bad])
    with pytest.raises(LLMInvalidResponseError) as exc:
        complete_structured(llm, MSGS, _Output, max_retries=0)
    assert "conf" in str(exc.value) or "confidence" in str(exc.value)


def test_connection_error_propagates_immediately_without_json_retries():
    class _DeadLLM:
        model_name = "dead"

        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            raise LLMConnectionError("provider down")

    llm = _DeadLLM()
    with pytest.raises(LLMConnectionError):
        complete_structured(llm, MSGS, _Output, max_retries=5)
    assert llm.calls == 1  # no pointless retries on a dead endpoint


def test_complete_structured_uses_mock_client_end_to_end():
    client = MockLLMClient()
    parsed, responses = complete_structured(client, MSGS, _Output, mock_key="RESEARCHER")
    assert parsed.evidence
    assert len(responses) == 1
