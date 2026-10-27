"""Optional NLI-based detector (experimental).

Idea: Natural-Language-Inference models are trained on exactly the
question we care about — given a premise, is a hypothesis ENTAILED,
CONTRADICTED, or NEUTRAL? We map:

    premise = SOURCE,  hypothesis = TARGET
    contradiction -> CONTRADICTS      entailment  -> SUPPORTS
    neutral       -> UNRELATED   (+ lexical overrides for the
                  remaining types: REDUNDANT / DEPENDS_ON / REFINES)

This detector is OPTIONAL and never selected by the factory default: it
adds a second model download and only improves recall for the 3-way
entailment cases. Use it explicitly if wanted:

    from latent_workspace.relations import NLIRelationDetector
    detector = NLIRelationDetector()
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models import RelationshipType
from .base import RelationDetector, RelationDetectorError, RelationPrediction
from .mock_detector import (
    _content_tokens,
    _DEPENDENCY_MARKERS,
    _jaccard,
    _round2,
)

# standard HF NLI label order for cross-encoder/nli-* checkpoints
_CONTRADICTION, _NEUTRAL, _ENTAILED = 0, 1, 2


class NLIRelationDetector(RelationDetector):
    """Cross-encoder NLI classifier with lexical post-processing."""

    name = "nli"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        model=None,  # injection point for tests (no download needed)
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.nli_model
        self.detector_label = f"nli:{self.model_name}"

        if model is not None:
            self._model = model
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RelationDetectorError(
                "sentence-transformers is required for the NLI detector"
            ) from exc
        try:
            self._model = CrossEncoder(
                self.model_name, device=device or settings.embedding_device
            )
        except Exception as exc:
            raise RelationDetectorError(
                f"Could not load NLI model '{self.model_name}' "
                f"({type(exc).__name__}): {exc}. First use downloads it; "
                "check your network or use the mock/llm detector instead."
            ) from exc

    def detect(self, source_text: str, target_text: str) -> RelationPrediction:
        src = (source_text or "").strip()
        tgt = (target_text or "").strip()
        if not src or not tgt:
            raise RelationDetectorError(
                "Both source and target texts are required for classification"
            )

        try:
            scores = self._model.predict([(src, tgt)])[0]  # premise=source
        except Exception as exc:
            raise RelationDetectorError(f"NLI prediction failed: {exc}") from exc

        try:
            p_contra, p_neutral, p_entail = [float(x) for x in scores]
        except (TypeError, ValueError) as exc:
            raise RelationDetectorError(f"Unexpected NLI output: {scores!r}") from exc

        src_set, tgt_set = set(_content_tokens(src)), set(_content_tokens(tgt))
        jaccard = _jaccard(src_set, tgt_set)
        src_lower = src.lower()

        # lexical overrides for the types NLI cannot express
        if jaccard >= 0.75:
            return self._predict(
                RelationshipType.REDUNDANT,
                _round2(max(jaccard, 0.8)),
                f"near-duplicate content (jaccard={jaccard:.2f})",
            )
        for marker in _DEPENDENCY_MARKERS:
            if marker in src_lower:
                return self._predict(
                    RelationshipType.DEPENDS_ON,
                    _round2(max(p_entail, 0.6)),
                    f"dependency phrase '{marker}' in source",
                )
        containment = (len(src_set & tgt_set) / len(tgt_set)) if tgt_set else 0.0
        if (
            containment >= 0.9
            and len(src) / max(len(tgt), 1) >= 1.4
            and p_entail >= 0.35
            and p_contra < 0.35
        ):
            return self._predict(
                RelationshipType.REFINES,
                _round2(max(p_entail, 0.7)),
                f"source extends a contained target (containment={containment:.2f})",
            )

        # NLI core mapping
        if p_contra >= 0.55:
            return self._predict(
                RelationshipType.CONTRADICTS,
                _round2(p_contra),
                f"NLI contradiction probability {p_contra:.2f}",
            )
        if p_entail >= 0.55:
            return self._predict(
                RelationshipType.SUPPORTS,
                _round2(p_entail),
                f"NLI entailment probability {p_entail:.2f}",
            )
        return self._predict(
            RelationshipType.UNRELATED,
            _round2(p_neutral),
            f"NLI neutral probability {p_neutral:.2f}",
        )

    def _predict(
        self,
        rel: RelationshipType,
        confidence: float,
        rationale: str,
    ) -> RelationPrediction:
        return RelationPrediction(
            relationship_type=rel,
            confidence=confidence,
            rationale=rationale,
            detector=self.detector_label,
        )
