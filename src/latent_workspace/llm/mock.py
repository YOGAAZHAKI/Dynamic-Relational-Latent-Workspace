"""Deterministic, fully offline LLM client.

Why it exists
-------------
Tests and demos must run without API keys, network or money, and must
give the *same* result every run (stable tests).

How it works
------------
1. ``complete()`` records the call (tests can inspect it).
2. If ``mock_key`` has a scripted reply in ``self.scripted`` -> use it
   (tests use this to feed invalid JSON and exercise error paths).
3. If ``mock_key`` is in ``fail_keys`` -> raise :class:`LLMConnectionError`
   (simulates an outage).
4. Otherwise generate a realistic, schema-valid reply for that agent role
   from the text it was given.

``mock_key`` is the agent role string (``"PLANNER"``, ``"RESEARCHER"``,
``"CRITIC"``, ``"SYNTHESIZER"``). Real clients ignore it.
"""

from __future__ import annotations

import json
import re
import time
from typing import Sequence

from .base import ChatMessage, LLMConnectionError, LLMResponse

_MAX_SNIPPET = 100  # chars of the task/query echoed into mock content


def _grab(text: str, label: str) -> str:
    """Extract the value after ``Label:`` in a prompt.

    Stops at the next section (a blank line, or the standard
    ``Retrieved workspace context`` header) or at the end of the text.
    Whitespace is squashed to one line so generated JSON stays valid.
    Limitation: a value itself must not contain a blank line.
    """
    match = re.search(
        rf"{re.escape(label)}:\s*(.*?)\s*"
        r"(?=\n\s*\n|\n\s*Retrieved workspace context|\Z)",
        text,
        flags=re.S,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _snippet(text: str) -> str:
    text = text or "the question"
    return text[:_MAX_SNIPPET] + ("..." if len(text) > _MAX_SNIPPET else "")


def _estimate_tokens(text: str) -> int:
    """Cheap deterministic proxy for real token counts (~4 chars/token)."""
    return max(1, len(text) // 4)


class MockLLMClient:
    """Offline stand-in for a real LLM."""

    model_name = "mock-llm"

    def __init__(
        self,
        scripted: dict[str, str] | None = None,
        fail_keys: Sequence[str] | tuple[str, ...] = (),
    ) -> None:
        self.scripted: dict[str, str] = dict(scripted or {})
        self.fail_keys: set[str] = set(fail_keys)
        self.calls: list[dict] = []  # inspection trail for tests

    # -- interface ------------------------------------------------------
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        mock_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        key = mock_key or "UNKNOWN"
        user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        self.calls.append(
            {
                "mock_key": key,
                "n_messages": len(messages),
                "user_text": user_text,
                "turns": len(messages),  # grows when JSON retry kicks in
            }
        )
        started = time.perf_counter()
        if key in self.fail_keys:
            raise LLMConnectionError(f"mock outage for key={key}")

        if key in self.scripted:
            text = self.scripted[key]
        else:
            text = self._generate(key, messages, user_text)
        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=_estimate_tokens(
                " ".join(m.content for m in messages)
            ),
            completion_tokens=_estimate_tokens(text),
            total_tokens=_estimate_tokens(text)
            + _estimate_tokens(" ".join(m.content for m in messages)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- canned generations --------------------------------------------
    def _generate(self, key: str, messages: Sequence[ChatMessage], user_text: str) -> str:
        if key == "PLANNER":
            return self._planner(_grab(user_text, "User query"))
        if key == "RESEARCHER":
            return self._researcher(_grab(user_text, "Task"))
        if key == "CRITIC":
            return self._critic(_grab(user_text, "Task"))
        if key == "SYNTHESIZER":
            return self._synthesizer(user_text)
        if key == "RELATION_DETECTOR":
            return self._relation_detector(user_text)
        # Unknown key: still produce parseable JSON so pipelines keep running.
        return json.dumps({"note": f"no mock content for key={key}"})

    def _relation_detector(self, user_text: str) -> str:
        """Canned structured reply for the LLM relation detector.

        Delegates to the rule-based detector so the LLM path behaves
        sensibly offline (imports inside the method to avoid an
        import-time cycle: llm -> relations -> llm).
        """
        from ..relations.mock_detector import MockRelationDetector

        source = _grab(user_text, "Source statement")
        target = _grab(user_text, "Target statement")
        try:
            prediction = MockRelationDetector().detect(source, target)
            payload = {
                "relationship": prediction.relationship_type.value,
                "confidence": prediction.confidence,
                "rationale": prediction.rationale,
            }
        except Exception:
            payload = {
                "relationship": "UNRELATED",
                "confidence": 0.5,
                "rationale": "mock fallback: could not apply rules",
            }
        return json.dumps(payload, indent=2)

    def _planner(self, query: str) -> str:
        q = _snippet(query)
        payload = {
            "subtasks": [
                {
                    "description": f"Gather key facts and definitions needed to answer: {q}",
                    "assigned_agent": "RESEARCHER",
                },
                {
                    "description": f"Test the main competing explanations regarding: {q}",
                    "assigned_agent": "RESEARCHER",
                },
                {
                    "description": f"Audit assumptions and hunt for counterevidence about: {q}",
                    "assigned_agent": "CRITIC",
                },
            ],
            "hypotheses": [
                {
                    "content": f"Working hypothesis A: a primary underlying factor drives the phenomena in '{q}'",
                    "confidence": 0.60,
                },
                {
                    "content": f"Working hypothesis B: an alternative contributing factor explains '{q}'",
                    "confidence": 0.45,
                },
            ],
        }
        return json.dumps(payload, indent=2)

    def _researcher(self, task: str) -> str:
        t = _snippet(task)
        payload = {
            "evidence": [
                {
                    "content": f"Reported measurements consistently show the pattern described in: {t}",
                    "information_type": "EVIDENCE",
                    "confidence": 0.78,
                    "sources": ["mock-source: dataset-A"],
                    "depends_on": ["accuracy of the underlying measurements"],
                },
                {
                    "content": f"Background fact: comparable cases have been documented for: {t}",
                    "information_type": "FACT",
                    "confidence": 0.70,
                    "sources": ["mock-source: reference-B"],
                    "depends_on": [],
                },
                {
                    "content": f"Additional indication: independent signals point the same way for: {t}",
                    "information_type": "EVIDENCE",
                    "confidence": 0.66,
                    "sources": [],
                    "depends_on": ["completeness of the collected signals"],
                },
            ]
        }
        return json.dumps(payload, indent=2)

    def _critic(self, task: str) -> str:
        t = _snippet(task)
        payload = {
            "critiques": [
                {
                    "content": f"Possible counterevidence: observed cases contradict the strong form of the claim in: {t}",
                    "information_type": "COUNTER_EVIDENCE",
                    "target": "primary hypothesis",
                    "confidence": 0.72,
                    "issues": ["possible contradiction", "small sample"],
                },
                {
                    "content": f"Refined conclusion: a hedged version of the claim for: {t} survives scrutiny",
                    "information_type": "REFINEMENT",
                    "target": "working hypotheses",
                    "confidence": 0.80,
                    "issues": ["overstated confidence"],
                },
            ]
        }
        return json.dumps(payload, indent=2)

    def _synthesizer(self, user_text: str) -> str:
        query = _grab(user_text, "Original query") or "the question"
        cited = re.findall(r"\[(iu_[0-9a-f]{8})\]", user_text)
        cites = ", ".join(f"[{cid}]" for cid in cited[:4]) if cited else "the available workspace units"
        return (
            f"Synthesized answer (mock): based on {len(cited) if cited else 'the'} "
            f"retrieved workspace unit(s), the evidence converges on a cautious "
            f"conclusion about: {_snippet(query)} "
            f"Supporting/contradicting units cited: {cites}."
        )
