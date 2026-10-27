"""Deterministic, rule-based relation detector (offline).

How it works
------------
A short, ordered rule stack over cheap lexical signals. Every prediction
carries a rationale naming the signal that fired — the same signals the
rules document, so behavior is auditable:

1. REDUNDANT      near-identical content (high token overlap, similar length)
2. CONTRADICTS    negation in exactly one side, or antonym words across
                  sides, with enough shared topic to be about the same thing
3. DEPENDS_ON     source contains a dependency phrase ("depends on", ...)
4. SUPPORTS       explicit support vocabulary ("confirms", "evidence", ...)
5. REFINES        source strictly extends a contained target (and used no
                  support vocabulary - an evidence statement that merely
                  contains a claim is support, not refinement)
6. SUPPORTS       weaker: containment without support words, or plain overlap
7. UNRELATED      nothing above matched

Confidence is a deterministic formula per rule (no randomness). Bare
content overlap alone yields a LOW-confidence SUPPORTS (~0.5) — below the
default 0.65 threshold — so coincidental word sharing does not create
edges. That is a deliberate safety property of the whole design.
"""

from __future__ import annotations

import re

from ..models import RelationshipType
from .base import RelationDetector, RelationDetectorError, RelationPrediction

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# words too common to carry topical meaning (kept out of overlap math)
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "it", "its", "that",
    "this", "these", "those", "with", "as", "at", "by", "from", "we",
    "they", "their", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "than", "then", "when", "while",
    "about", "into", "over", "under", "between", "during", "after",
    "before", "more", "most", "some", "any", "all", "each", "such",
}

_NEGATION_RE = re.compile(
    r"\b(not|never|no|cannot|without|neither|nobody|nothing|nowhere|fails?)\b"
    r"|n't\b",
    re.I,
)

_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("rise", "fall"), ("rose", "fell"), ("rose", "dropped"),
    ("increase", "decrease"), ("increased", "decreased"),
    ("higher", "lower"), ("grew", "shrank"), ("expand", "shrink"),
    ("expanded", "contracted"), ("gain", "loss"), ("profit", "loss"),
    ("safe", "unsafe"), ("effective", "ineffective"), ("boost", "cut"),
    ("surge", "plunge"), ("outperform", "underperform"),
    ("supports", "refutes"), ("prevents", "causes"),
)

_SUPPORT_MARKERS = {
    "confirms", "confirmed", "supports", "support", "consistent",
    "therefore", "thus", "hence", "evidence", "demonstrates",
    "demonstrate", "shows", "show", "proves", "prove", "because",
    "indicates", "indicate", "implies", "affirms", "validates",
}

_DEPENDENCY_MARKERS = (
    "depends on", "depends upon", "requires", "require", "presupposes",
    "assumes", "assume", "relies on", "rests on", "needs", "contingent",
    "based on",
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _round2(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 2)


class MockRelationDetector(RelationDetector):
    """Rule-based detector. Same input -> same output, always."""

    name = "mock"
    detector_label = "mock-rules"

    # rule knobs (class attrs so behavior is easy to audit and teach)
    REDUNDANT_JACCARD = 0.75
    LENGTH_RATIO_MAX = 1.6
    CONTRADICT_MIN_JACCARD = 0.15
    REFINE_CONTAINMENT = 0.9
    REFINE_LENGTH_RATIO = 1.4
    WEAK_SUPPORT_JACCARD = 0.20

    def detect(self, source_text: str, target_text: str) -> RelationPrediction:
        src = (source_text or "").strip()
        tgt = (target_text or "").strip()
        if not src or not tgt:
            raise RelationDetectorError(
                "Both source and target texts are required for classification"
            )

        src_tokens = _content_tokens(src)
        tgt_tokens = _content_tokens(tgt)
        src_set, tgt_set = set(src_tokens), set(tgt_tokens)
        jaccard = _jaccard(src_set, tgt_set)
        inter = len(src_set & tgt_set)

        # -- 1. REDUNDANT ----------------------------------------------------
        min_len = min(len(src_tokens), len(tgt_tokens))
        max_len = max(len(src_tokens), len(tgt_tokens))
        length_ratio = (max_len / min_len) if min_len else float("inf")
        if jaccard >= self.REDUNDANT_JACCARD and length_ratio <= self.LENGTH_RATIO_MAX:
            return self._predict(
                RelationshipType.REDUNDANT,
                _round2(0.85 + 0.10 * jaccard),
                f"near-duplicate content (jaccard={jaccard:.2f}, "
                f"length ratio={length_ratio:.2f})",
            )

        # -- 2. CONTRADICTS --------------------------------------------------
        neg_src = bool(_NEGATION_RE.search(src))
        neg_tgt = bool(_NEGATION_RE.search(tgt))
        antonym = self._antonym_across(src_set, tgt_set)
        if jaccard >= self.CONTRADICT_MIN_JACCARD or antonym:
            negation_signal = (neg_src != neg_tgt) and inter >= 1
            antonym_signal = antonym is not None and inter >= 1
            if negation_signal or antonym_signal:
                signals = []
                if negation_signal:
                    signals.append(
                        f"negation in {'source' if neg_src else 'target'} only"
                    )
                if antonym_signal and antonym is not None:
                    signals.append(f"antonym pair '{antonym[0]}' vs '{antonym[1]}'")
                conf = 0.70 + 0.20 * min(jaccard, 1.0) + (0.05 if antonym_signal else 0)
                return self._predict(
                    RelationshipType.CONTRADICTS,
                    _round2(conf),
                    f"{'; '.join(signals)} with shared topic (jaccard={jaccard:.2f})",
                )

        # -- 3. DEPENDS_ON -----------------------------------------------------
        src_lower = src.lower()
        for marker in _DEPENDENCY_MARKERS:
            if marker in src_lower:
                return self._predict(
                    RelationshipType.DEPENDS_ON,
                    _round2(0.70 + 0.10 * min(jaccard, 1.0)),
                    f"source uses dependency phrase '{marker}' "
                    f"(jaccard={jaccard:.2f})",
                )

        # -- 4. SUPPORTS via explicit support vocabulary -------------------------
        support_words = (src_set | tgt_set) & _SUPPORT_MARKERS
        if support_words and jaccard >= self.CONTRADICT_MIN_JACCARD:
            return self._predict(
                RelationshipType.SUPPORTS,
                _round2(0.65 + 0.25 * min(jaccard, 1.0)),
                f"support vocabulary {sorted(support_words)[:3]} with shared "
                f"topic (jaccard={jaccard:.2f})",
            )

        # -- 5. REFINES (source strictly extends a contained target) -------------
        containment = (inter / len(tgt_set)) if tgt_set else 0.0
        char_ratio = len(src) / max(len(tgt), 1)
        if (
            tgt_set
            and containment >= self.REFINE_CONTAINMENT
            and char_ratio >= self.REFINE_LENGTH_RATIO
            and inter >= 2
        ):
            return self._predict(
                RelationshipType.REFINES,
                _round2(0.75 + 0.10 * containment),
                f"source contains the target and adds detail "
                f"(containment={containment:.2f}, length ratio={char_ratio:.2f})",
            )

        # -- 6. SUPPORTS (weaker forms) --------------------------------------------
        if tgt_set and containment >= self.REFINE_CONTAINMENT and inter >= 2:
            # the target extends the narrower source: consistent, but weaker
            return self._predict(
                RelationshipType.SUPPORTS,
                _round2(0.50 + 0.10 * containment),
                "source is consistent with the more specific target "
                f"(containment={containment:.2f})",
            )
        if inter >= 2 and jaccard >= self.WEAK_SUPPORT_JACCARD:
            return self._predict(
                RelationshipType.SUPPORTS,
                _round2(0.50 + 0.20 * jaccard),
                f"shared content words only, no typed signal "
                f"(jaccard={jaccard:.2f})",
            )

        # -- 7. UNRELATED -----------------------------------------------------------
        return self._predict(
            RelationshipType.UNRELATED,
            _round2(0.90 - 0.40 * jaccard),
            f"no typed signal and little shared content (jaccard={jaccard:.2f})",
        )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _antonym_across(src: set[str], tgt: set[str]) -> tuple[str, str] | None:
        for a, b in _ANTONYM_PAIRS:
            a_in_src, b_in_src = a in src, b in src
            a_in_tgt, b_in_tgt = a in tgt, b in tgt
            if (a_in_src and b_in_tgt) or (b_in_src and a_in_tgt):
                return (a, b)
        return None

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
