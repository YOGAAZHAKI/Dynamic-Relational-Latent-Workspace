"""Tests for the evaluation harness + baselines (chunk 9).

Offline end-to-end over the deterministic mock stack. The harness is
tested both at the metric level (hand-built EvalRun records) and as a
full experiment over all default systems.
"""

from __future__ import annotations

import pytest

from latent_workspace.config import Settings
from latent_workspace.evaluation import (
    DEFAULT_CASES,
    EvalCase,
    EvalRun,
    EvalUnit,
    EvaluationHarness,
    evaluate_run,
)


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


# ---------------------------------------------------------------------------
# dataset sanity
# ---------------------------------------------------------------------------

def test_default_cases_are_unique_and_have_conflicts():
    ids = [case.id for case in DEFAULT_CASES]
    assert len(ids) == len(set(ids))
    for case in DEFAULT_CASES:
        assert case.relevant_units
        assert case.conflict_units
        assert case.key_points


def test_case_marker_helpers():
    case = DEFAULT_CASES[0]
    relevant = case.relevant_units
    conflicts = case.conflict_units
    assert conflicts
    assert all(case.is_conflict(u.content) for u in conflicts)
    assert all(case.is_relevant(u.content) for u in relevant)


# ---------------------------------------------------------------------------
# metrics on controlled records
# ---------------------------------------------------------------------------

def _toy_case() -> EvalCase:
    return EvalCase(
        id="toy",
        query="Did alpha rise?",
        units=[
            EvalUnit(content="Alpha rose 10% in Q1."),
            EvalUnit(
                content="Alpha did not rise in Q1.",
            ),
            EvalUnit(content="The football team won."),
        ],
        relevant_markers=["alpha"],
        conflict_markers=["did not rise"],
        key_points=["alpha rose"],
    )


def test_metrics_reward_relevant_conflict_and_citations():
    case = _toy_case()
    contents = {u.content: f"iu_{i:08x}" for i, u in enumerate(case.units, start=1)}
    ids = {content: uid for content, uid in contents.items()}
    run = EvalRun(
        system="x",
        case_id=case.id,
        query=case.query,
        answer=f"Alpha rose, per [iu_00000001]; counter-evidence: [iu_00000002].",
        retrieved_unit_ids=list(ids.values()),
        retrieved_contents=list(contents),
        workspace_contents={uid: content for content, uid in ids.items()},
        context_tokens=50,
        total_tokens=100,
    )
    metrics = evaluate_run(case, run)
    assert metrics.retrieval_recall == 1.0
    assert metrics.retrieval_precision == pytest.approx(2 / 3, abs=1e-4)
    assert metrics.conflict_recall == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0  # both relevant units are cited
    assert metrics.answer_key_point_recall == 1.0
    assert metrics.composite > 0.8


def test_metrics_penalise_missing_conflict_and_bad_citations():
    case = _toy_case()
    run = EvalRun(
        system="x",
        case_id=case.id,
        query=case.query,
        answer="No evidence at all [iu_deadbeef].",
        retrieved_contents=["Alpha rose 10% in Q1."],
        workspace_contents={"iu_00000001": "Alpha rose 10% in Q1."},
    )
    metrics = evaluate_run(case, run)
    assert metrics.conflict_recall == 0.0
    assert metrics.citation_precision == 0.0
    assert metrics.grounded_key_point_recall == 0.0
    assert metrics.composite < 0.8


def test_metrics_when_case_has_no_conflict():
    case = EvalCase(
        id="no_conflict",
        query="alpha?",
        units=[EvalUnit(content="Alpha rose.")],
        relevant_markers=["alpha"],
        key_points=["alpha"],
    )
    run = EvalRun(
        system="x",
        case_id=case.id,
        query=case.query,
        retrieved_contents=["Alpha rose."],
        workspace_contents={},
    )
    assert evaluate_run(case, run).conflict_recall is None


# ---------------------------------------------------------------------------
# full harness
# ---------------------------------------------------------------------------

def test_harness_runs_all_systems_offline(settings):
    report = EvaluationHarness(settings=settings).run()

    assert report.mode == "mock (offline)"
    names = [s.system for s in report.systems]
    assert names == [
        "A: llm_only",
        "B: flat_context",
        "C: vector_rag",
        "D: drlw",
        "ablation: no_expansion",
    ]
    assert {r.case_id for r in report.results} == {c.id for c in DEFAULT_CASES}

    by_name = {s.system: s for s in report.systems}
    assert by_name["A: llm_only"].metrics["retrieval_recall"] == 0.0
    assert by_name["B: flat_context"].metrics["retrieval_recall"] == 1.0
    assert by_name["D: drlw"].metrics["retrieval_recall"] > 0.0
    assert by_name["D: drlw"].metrics["conflict_recall"] == 1.0
    assert by_name["D: drlw"].mean_total_tokens > 0
    assert by_name["D: drlw"].errors == 0
    # the typed graph + expansion must beat the no-expansion ablation
    assert (
        by_name["D: drlw"].composite
        > by_name["ablation: no_expansion"].composite
    )


def test_report_serialises_and_renders_markdown(settings):
    report = EvaluationHarness(settings=settings).run()
    payload = report.to_dict()
    assert isinstance(payload, dict)
    assert len(payload["systems"]) == len(report.systems)

    markdown = report.to_markdown()
    assert "# DRLW evaluation report" in markdown
    assert "D: drlw" in markdown
    assert "Per-case composite" in markdown
    assert report.ranking()[0].composite >= report.ranking()[-1].composite


def test_harness_is_deterministic_in_mock_mode(settings):
    first = EvaluationHarness(settings=settings).run()
    second = EvaluationHarness(settings=settings).run()
    for a, b in zip(first.systems, second.systems):
        assert a.system == b.system
        assert a.metrics == b.metrics


def test_harness_rejects_duplicate_case_ids(settings):
    case = DEFAULT_CASES[0]
    with pytest.raises(ValueError, match="unique"):
        EvaluationHarness(cases=[case, case], settings=settings)
