"""Orchestration package: query -> multi-agent run -> grounded answer."""

from __future__ import annotations

from .orchestrator import (
    Orchestrator,
    OrchestratorError,
    build_orchestrator,
    extract_citations,
    fallback_answer,
)

__all__ = [
    "Orchestrator",
    "OrchestratorError",
    "build_orchestrator",
    "extract_citations",
    "fallback_answer",
]
