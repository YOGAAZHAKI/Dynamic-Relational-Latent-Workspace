"""Chunk 4 demo: relationship detection over candidate text pairs.

    python scripts/demo_relations.py          # rule-based detector (offline)
    python scripts/demo_relations.py --llm    # LLM-shaped path via mock LLM

For each pair it shows: predicted relationship, confidence, whether an
edge would be created at the configured threshold, and the rationale.
"""

from __future__ import annotations

import argparse
import sys

from latent_workspace.config import get_settings
from latent_workspace.llm import MockLLMClient
from latent_workspace.relations import (
    LLMRelationDetector,
    MockRelationDetector,
    RelationDetectorError,
    build_relation_detector,
)

PAIRS = [
    (
        "SUPPORTS",
        "Regional data confirms that industrial electricity prices increased sharply.",
        "Electricity prices rose for industrial users.",
    ),
    (
        "CONTRADICTS",
        "The study shows that vaccines do not prevent infection.",
        "Vaccines prevent infection.",
    ),
    (
        "REFINES",
        "Electricity prices rose approximately 12% in the industrial sector during Q3 2025.",
        "Electricity prices rose.",
    ),
    (
        "DEPENDS_ON",
        "This conclusion depends on the assumption that electricity demand data is accurate.",
        "Electricity demand data is accurate for Q3.",
    ),
    (
        "REDUNDANT",
        "Oil prices fell sharply on Tuesday.",
        "Oil prices fell sharply on Tuesday.",
    ),
    (
        "UNRELATED",
        "The football team won the championship final.",
        "Telescope images revealed a distant galaxy cluster.",
    ),
    (
        "weak overlap (safety demo)",
        "The committee met in March to discuss budget allocations.",
        "March is when the budget numbers were reviewed by auditors.",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="DRLW relation detection demo")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use the LLM detector backed by the offline mock LLM",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.llm:
        detector = LLMRelationDetector(MockLLMClient(), settings)
    elif settings.mock_mode:
        detector = MockRelationDetector()
    else:
        detector = build_relation_detector(settings=settings)

    threshold = settings.relation_threshold
    print(f"detector   : {getattr(detector, 'detector_label', detector.name)}")
    print(f"threshold  : {threshold}  (edges need a typed relation above this)")
    print(f"pairs      : {len(PAIRS)}\n")

    edges = 0
    for label, source, target in PAIRS:
        try:
            pred = detector.detect(source, target)
        except RelationDetectorError as exc:
            print(f"[{label}] FAILED: {exc}\n")
            continue
        edge = pred.to_edge("iu_source", "iu_target", threshold=threshold)
        verdict = "EDGE CREATED" if edge else "filtered out (no edge)"
        if edge:
            edges += 1
        print(f"--- {label}")
        print(f"    source     : {source}")
        print(f"    target     : {target}")
        print(f"    prediction : {pred.relationship_type.value} "
              f"(confidence {pred.confidence:.2f})")
        print(f"    decision   : {verdict}")
        print(f"    rationale  : {pred.rationale}")
        if edge:
            print(f"    edge       : {edge.direction_text()}")
        print()

    print(f"edges created at threshold {threshold}: {edges} / {len(PAIRS)}")
    if any(p[0].startswith("weak") for p in PAIRS):
        print("(the 'weak overlap' pair is a SUPPORTS prediction kept BELOW the "
              "threshold on purpose)")
    print("OK - chunk 4 demo finished with zero API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
