"""LLM-based relation detector (the default for live mode).

Asks the chat model to *read* both statements and return a strict JSON
``{relationship, confidence, rationale}``, validated by pydantic through
the same ``complete_structured`` retry machinery the agents use.

Failures (garbage replies, invalid relation types, outages) are wrapped
in :class:`RelationDetectorError` so the workspace can skip a pair
instead of crashing a whole run.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..llm.base import ChatMessage, LLMClient, LLMError
from ..llm.structured import complete_structured
from ..models import RelationshipType
from .base import RelationDetector, RelationDetectorError, RelationPrediction

RELATION_SYSTEM = """You are a RELATION DETECTOR in a multi-agent reasoning system.

You are given two statements:
  SOURCE - the newer statement (often evidence or a critique)
  TARGET - an existing statement it was compared against

Decide how SOURCE relates to TARGET. The answer reads
"SOURCE <relationship> TARGET":

SUPPORTS     source gives evidence for / backs up target
CONTRADICTS  source conflicts with or refutes target
REFINES      source makes target more precise or better scoped
DEPENDS_ON   source only holds if target holds (source needs target)
REDUNDANT    source and target say essentially the same thing
UNRELATED    no meaningful semantic relation

CRITICAL RULES:
- Wording similarity is NOT support. "vaccines prevent disease" and
  "vaccines cause disease" are highly similar yet CONTRADICTS.
- Judge the CLAIMS, not the topic. Same topic + opposing claims = CONTRADICTS.
- Prefer UNRELATED when the statements can both be true independently.
- Confidence reflects how sure you are of the TYPE (0.0-1.0).

Respond with ONLY a single JSON object, no markdown fences, no prose:
{"relationship": "SUPPORTS|CONTRADICTS|REFINES|DEPENDS_ON|REDUNDANT|UNRELATED",
 "confidence": 0.0,
 "rationale": "<one short sentence explaining the judgment>"}
"""

RELATION_USER = """Source statement:
__SOURCE__

Target statement:
__TARGET__
"""


class RelationSpec(BaseModel):
    """JSON schema the LLM's reply must satisfy."""

    relationship: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class LLMRelationDetector(RelationDetector):
    name = "llm"

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    @property
    def detector_label(self) -> str:
        return f"llm:{getattr(self.llm, 'model_name', 'unknown')}"

    def detect(self, source_text: str, target_text: str) -> RelationPrediction:
        src = (source_text or "").strip()
        tgt = (target_text or "").strip()
        if not src or not tgt:
            raise RelationDetectorError(
                "Both source and target texts are required for classification"
            )

        user = (
            RELATION_USER
            .replace("__SOURCE__", src)
            .replace("__TARGET__", tgt)
        )
        messages = [
            ChatMessage(role="system", content=RELATION_SYSTEM),
            ChatMessage(role="user", content=user),
        ]
        try:
            spec, _responses = complete_structured(
                self.llm,
                messages,
                RelationSpec,
                mock_key="RELATION_DETECTOR",
                max_retries=self.settings.llm_json_retries,
            )
        except LLMError as exc:
            raise RelationDetectorError(
                f"LLM relation detection failed ({type(exc).__name__}) for pair "
                f"(source={src[:60]!r}, target={tgt[:60]!r}): {exc}"
            ) from exc

        return RelationPrediction(
            relationship_type=spec.relationship,
            confidence=spec.confidence,
            rationale=spec.rationale.strip(),
            detector=self.detector_label,
        )
