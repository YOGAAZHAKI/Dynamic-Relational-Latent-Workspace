"""Evaluation harness (chunk 9): baselines A-D + no-expansion ablation.

    from latent_workspace.evaluation import EvaluationHarness
    report = EvaluationHarness().run()
    Path("results/eval.md").write_text(report.to_markdown())

Everything runs offline while ``DLRW_MOCK_MODE=true``.
"""

from __future__ import annotations

from .dataset import DEFAULT_CASES, EvalCase, EvalUnit
from .harness import (
    CaseResult,
    EvaluationHarness,
    EvaluationReport,
    SystemSummary,
)
from .metrics import METRIC_NAMES, CaseMetrics, evaluate_run, mean
from .systems import (
    DirectLLMSystem,
    EvalRun,
    EvalSystem,
    FlatContextSystem,
    NoExpansionSystem,
    OrchestratorSystem,
    VectorRAGSystem,
    default_systems,
)

__all__ = [
    "DEFAULT_CASES",
    "METRIC_NAMES",
    "CaseMetrics",
    "CaseResult",
    "DirectLLMSystem",
    "EvalCase",
    "EvalRun",
    "EvalSystem",
    "EvalUnit",
    "EvaluationHarness",
    "EvaluationReport",
    "FlatContextSystem",
    "NoExpansionSystem",
    "OrchestratorSystem",
    "SystemSummary",
    "VectorRAGSystem",
    "default_systems",
    "evaluate_run",
    "mean",
]
