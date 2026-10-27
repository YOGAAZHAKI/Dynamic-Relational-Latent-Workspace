"""Deterministic, offline metrics for the evaluation harness (chunk 9).

No judge model is involved. Each metric is a transparent string/ID check
against the case's own markers and key points:

- retrieval recall / precision (did the right units reach the context?)
- conflict recall (was counter-evidence retrieved, not hidden?)
- citation precision / recall (were answers grounded in real units?)
- key-point recall, both answer-only and "grounded" (answer OR cited unit)
- a weighted composite in [0, 1]

Matching is case-insensitive substring matching after whitespace
normalisation. This is intentionally simple and reproducible; it is not
a substitute for human evaluation of factual correctness.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..orchestrator import extract_citations
from .dataset import EvalCase
from .systems import EvalRun

_WS_RE = re.compile(r"\s+")

METRIC_NAMES: tuple[str, ...] = (
    "retrieval_recall",
    "retrieval_precision",
    "conflict_recall",
    "citation_precision",
    "citation_recall",
    "answer_key_point_recall",
    "grounded_key_point_recall",
    "composite",
)


class CaseMetrics(BaseModel):
    """Scores for one (system, case) pair."""

    retrieval_recall: float = Field(ge=0.0, le=1.0)
    retrieval_precision: float = Field(ge=0.0, le=1.0)
    conflict_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    answer_key_point_recall: float = Field(ge=0.0, le=1.0)
    grounded_key_point_recall: float = Field(ge=0.0, le=1.0)
    composite: float = Field(ge=0.0, le=1.0)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def _covers(text: str, phrase: str) -> bool:
    phrase = _norm(phrase)
    return bool(phrase) and phrase in _norm(text)


def _ratio(hit: int, total: int) -> float:
    return round(hit / total, 4) if total else 0.0


def evaluate_run(case: EvalCase, run: EvalRun) -> CaseMetrics:
    """Score one system run against one case's expectations."""
    case_contents = {u.content for u in case.units}
    relevant_contents = [u.content for u in case.relevant_units]
    conflict_contents = [u.content for u in case.conflict_units]

    retrieved_dataset = [
        content for content in run.retrieved_contents if content in case_contents
    ]
    retrieved_relevant = [
        content for content in retrieved_dataset if case.is_relevant(content)
    ]
    retrieved_conflicts = [
        content for content in retrieved_dataset if case.is_conflict(content)
    ]

    retrieval_recall = _ratio(len(retrieved_relevant), len(relevant_contents))
    retrieval_precision = _ratio(len(retrieved_relevant), len(retrieved_dataset))
    conflict_recall = (
        _ratio(len(retrieved_conflicts), len(conflict_contents))
        if conflict_contents
        else None
    )

    citations = extract_citations(run.answer)
    valid_citations = [
        cid for cid in citations if cid in run.workspace_contents
    ]
    citation_precision = _ratio(len(valid_citations), len(citations))

    cited_relevant = {
        cid
        for cid in valid_citations
        if case.is_relevant(run.workspace_contents[cid])
    }
    citation_recall = _ratio(len(cited_relevant), len(relevant_contents))

    answer_hits = sum(1 for point in case.key_points if _covers(run.answer, point))
    answer_key_point_recall = _ratio(answer_hits, len(case.key_points))

    cited_text = " ".join(run.workspace_contents[cid] for cid in valid_citations)
    grounded_text = f"{run.answer} {cited_text}"
    grounded_hits = sum(
        1 for point in case.key_points if _covers(grounded_text, point)
    )
    grounded_key_point_recall = _ratio(grounded_hits, len(case.key_points))

    conflict_component = 1.0 if conflict_recall is None else conflict_recall
    composite = round(
        0.35 * retrieval_recall
        + 0.15 * retrieval_precision
        + 0.20 * conflict_component
        + 0.20 * grounded_key_point_recall
        + 0.10 * citation_precision,
        4,
    )

    return CaseMetrics(
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
        conflict_recall=conflict_recall,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        answer_key_point_recall=answer_key_point_recall,
        grounded_key_point_recall=grounded_key_point_recall,
        composite=composite,
    )


def mean(values: list[float | None]) -> float:
    """Mean of the non-None values (0.0 when nothing is present)."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return 0.0
    return round(sum(present) / len(present), 4)
