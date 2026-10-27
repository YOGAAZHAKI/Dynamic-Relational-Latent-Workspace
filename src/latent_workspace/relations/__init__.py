"""Relations package: detector interface + implementations + factory.

    from latent_workspace.relations import build_relation_detector
    detector = build_relation_detector()            # mock or LLM, per settings
    pred = detector.detect(source_text, target_text)
    edge = pred.to_edge(source_id, target_id, threshold=0.65)  # None = filtered
"""

from __future__ import annotations

from .base import (
    RelationDetector,
    RelationDetectorError,
    RelationPrediction,
)
from .llm_detector import LLMRelationDetector, RelationSpec
from .mock_detector import MockRelationDetector


def build_relation_detector(llm=None, settings=None) -> RelationDetector:
    """Default detector: rule-based in mock mode, LLM-based otherwise.

    The NLI detector is never the default (extra model download); use it
    explicitly via ``NLIRelationDetector`` if wanted.
    """
    from ..config import get_settings

    s = settings or get_settings()
    if s.mock_mode:
        return MockRelationDetector()
    if llm is None:
        from ..llm import build_llm

        llm = build_llm(s)
    return LLMRelationDetector(llm, s)


__all__ = [
    "LLMRelationDetector",
    "MockRelationDetector",
    "NLIRelationDetector",
    "RelationDetector",
    "RelationDetectorError",
    "RelationPrediction",
    "RelationSpec",
    "build_relation_detector",
]


def __getattr__(name: str):
    """Lazy import for the optional NLI detector (keeps imports cheap)."""
    if name == "NLIRelationDetector":
        from .nli_detector import NLIRelationDetector

        return NLIRelationDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
