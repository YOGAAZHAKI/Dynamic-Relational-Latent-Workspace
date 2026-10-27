"""Evaluation harness: run A-D + ablation over all cases (chunk 9).

Usage::

    from latent_workspace.evaluation import EvaluationHarness
    report = EvaluationHarness().run()
    print(report.to_markdown())

The harness is offline by default (settings mock mode). A failure in one
system/case is recorded as a zero-score run with an error message; it
never aborts the whole experiment.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import time
import uuid

from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..models import utc_now
from .dataset import DEFAULT_CASES, EvalCase
from .metrics import METRIC_NAMES, CaseMetrics, evaluate_run, mean
from .systems import EvalRun, EvalSystem, default_systems


@contextlib.contextmanager
def _deterministic_ids(seed: str = "drlw-evaluation"):
    """Make ``new_id`` deterministic for the duration of a harness run.

    Unit ids are random by design in the live system, but tie-breaking in
    retrieval is id-ordered. For reproducible experiments we substitute a
    fixed id sequence; content, scores and ordering then match run-to-run.
    """
    original = uuid.uuid4
    counter = itertools.count()

    def _fake_uuid() -> uuid.UUID:
        digest = hashlib.sha1(
            f"{seed}:{next(counter)}".encode("utf-8")
        ).hexdigest()
        return uuid.UUID(digest[:32])

    uuid.uuid4 = _fake_uuid  # type: ignore[assignment]
    try:
        yield
    finally:
        uuid.uuid4 = original  # type: ignore[assignment]


class CaseResult(BaseModel):
    """One system run plus its scores."""

    system: str
    case_id: str
    run: EvalRun
    metrics: CaseMetrics


class SystemSummary(BaseModel):
    """Aggregated metrics for one system across all cases."""

    system: str
    description: str
    cases: int
    errors: int
    metrics: dict[str, float] = Field(default_factory=dict)
    mean_context_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    mean_nodes: float = 0.0
    mean_edges: float = 0.0

    @property
    def composite(self) -> float:
        return self.metrics.get("composite", 0.0)


class EvaluationReport(BaseModel):
    """Complete, serialisable result of one harness run."""

    created_at: str
    mode: str
    cases: list[str]
    systems: list[SystemSummary]
    results: list[CaseResult]

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def ranking(self) -> list[SystemSummary]:
        return sorted(self.systems, key=lambda s: (-s.composite, s.system))

    def case_results(self, system: str) -> list[CaseResult]:
        return [r for r in self.results if r.system == system]

    def to_markdown(self) -> str:
        lines = [
            "# DRLW evaluation report",
            "",
            f"- created: {self.created_at}",
            f"- mode: **{self.mode}**",
            f"- cases: {len(self.cases)} ({', '.join(self.cases)})",
            "",
            "## System comparison (mean across cases)",
            "",
            "| system | retrieval R | retrieval P | conflict R | citation P | "
            "key points (grounded) | context tok | total tok | latency ms | composite |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in self.ranking():
            m = summary.metrics
            conflict = m.get("conflict_recall", 0.0)
            lines.append(
                f"| {summary.system} "
                f"| {m.get('retrieval_recall', 0.0):.2f} "
                f"| {m.get('retrieval_precision', 0.0):.2f} "
                f"| {conflict:.2f} "
                f"| {m.get('citation_precision', 0.0):.2f} "
                f"| {m.get('grounded_key_point_recall', 0.0):.2f} "
                f"| {summary.mean_context_tokens:.0f} "
                f"| {summary.mean_total_tokens:.0f} "
                f"| {summary.mean_latency_ms:.0f} "
                f"| **{summary.composite:.2f}** |"
            )
        lines.extend(
            [
                "",
                "## Per-case composite scores",
                "",
                "| case | " + " | ".join(s.system for s in self.ranking()) + " |",
                "|---|" + "---:|" * len(self.systems),
            ]
        )
        by_case: dict[str, dict[str, float]] = {case: {} for case in self.cases}
        for result in self.results:
            by_case.setdefault(result.case_id, {})[result.system] = (
                result.metrics.composite
            )
        for case in self.cases:
            scores = by_case.get(case, {})
            cells = " | ".join(
                f"{scores.get(s.system, 0.0):.2f}" for s in self.ranking()
            )
            lines.append(f"| {case} | {cells} |")

        error_systems = [s for s in self.systems if s.errors]
        if error_systems:
            lines.extend(["", "## Recorded errors", ""])
            for summary in error_systems:
                lines.append(
                    f"- {summary.system}: {summary.errors} error(s) across cases"
                )
        lines.extend(
            [
                "",
                "> Edges and contexts are predicted relationships with confidence; "
                "these scores measure retrieval/grounding behaviour, not factual truth.",
                "",
            ]
        )
        return "\n".join(lines)


class EvaluationHarness:
    """Compare systems over a deterministic set of cases."""

    def __init__(
        self,
        cases: list[EvalCase] | tuple[EvalCase, ...] | None = None,
        settings: Settings | None = None,
        systems: list[EvalSystem] | None = None,
        *,
        deterministic_ids: bool = True,
    ) -> None:
        self.cases = list(cases) if cases is not None else list(DEFAULT_CASES)
        if not self.cases:
            raise ValueError("at least one evaluation case is required")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique")
        self.settings = settings or get_settings()
        self.systems = systems or default_systems(self.settings)
        self.deterministic_ids = deterministic_ids

    def run(self) -> EvaluationReport:
        results: list[CaseResult] = []
        context = (
            _deterministic_ids() if self.deterministic_ids else contextlib.nullcontext()
        )
        with context:
            for system in self.systems:
                for case in self.cases:
                    result = self._run_one(system, case)
                    results.append(result)
        return EvaluationReport(
            created_at=utc_now().isoformat(),
            mode="mock (offline)" if self.settings.is_mock else "live",
            cases=[case.id for case in self.cases],
            systems=self._summaries(results),
            results=results,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _run_one(self, system: EvalSystem, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        try:
            run = system.run_case(case)
        except Exception as exc:  # noqa: BLE001 - one broken system is data
            run = EvalRun(
                system=system.name,
                case_id=case.id,
                query=case.query,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                errors=[f"system crashed ({type(exc).__name__}): {exc}"],
            )
        metrics = evaluate_run(case, run)
        return CaseResult(
            system=system.name,
            case_id=case.id,
            run=run,
            metrics=metrics,
        )

    def _summaries(self, results: list[CaseResult]) -> list[SystemSummary]:
        descriptions = {s.name: s.description for s in self.systems}
        summaries: list[SystemSummary] = []
        for system in self.systems:
            subset = [r for r in results if r.system == system.name]
            if not subset:
                continue
            summary = SystemSummary(
                system=system.name,
                description=descriptions.get(system.name, ""),
                cases=len(subset),
                errors=sum(len(r.run.errors) for r in subset),
                metrics={
                    name: mean(
                        [getattr(r.metrics, name) for r in subset]
                    )
                    for name in METRIC_NAMES
                },
                mean_context_tokens=mean(
                    [float(r.run.context_tokens) for r in subset]
                ),
                mean_total_tokens=mean(
                    [float(r.run.total_tokens) for r in subset]
                ),
                mean_latency_ms=mean(
                    [float(r.run.latency_ms) for r in subset]
                ),
                mean_nodes=mean([float(r.run.nodes) for r in subset]),
                mean_edges=mean([float(r.run.edges) for r in subset]),
            )
            summaries.append(summary)
        return summaries
