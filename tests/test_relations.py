"""Tests for the relation detection layer (Chunk 4).

Curated pairs cover all six relationship types plus the safety rules
(UNRELATED never stored; weak bare-overlap SUPPORTS stays below the
default threshold).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from latent_workspace.config import Settings
from latent_workspace.llm import MockLLMClient
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
    RelationshipType,
)
from latent_workspace.relations import (
    LLMRelationDetector,
    MockRelationDetector,
    RelationDetectorError,
    RelationPrediction,
    build_relation_detector,
)

# ---------------------------------------------------------------------------
# curated pairs: (source, target, expected_type, min_confidence)
# ---------------------------------------------------------------------------
PAIRS = [
    (
        "Regional data confirms that industrial electricity prices increased sharply.",
        "Electricity prices rose for industrial users.",
        RelationshipType.SUPPORTS,
        0.65,
    ),
    (
        "The study shows that vaccines do not prevent infection.",
        "Vaccines prevent infection.",
        RelationshipType.CONTRADICTS,
        0.70,
    ),
    (
        "Electricity prices rose approximately 12% in the industrial sector during Q3 2025.",
        "Electricity prices rose.",
        RelationshipType.REFINES,
        0.75,
    ),
    (
        "This conclusion depends on the assumption that electricity demand data is accurate.",
        "Electricity demand data is accurate for Q3.",
        RelationshipType.DEPENDS_ON,
        0.70,
    ),
    (
        "Oil prices fell sharply on Tuesday.",
        "Oil prices fell sharply on Tuesday.",
        RelationshipType.REDUNDANT,
        0.90,
    ),
    (
        "Committee notes were published online yesterday morning.",
        "Committee notes were published online yesterday.",
        RelationshipType.REDUNDANT,
        0.80,
    ),
    (
        "The football team won the championship final.",
        "Telescope images revealed a distant galaxy cluster.",
        RelationshipType.UNRELATED,
        0.80,
    ),
]


@pytest.fixture
def detector():
    return MockRelationDetector()


@pytest.fixture
def settings():
    return Settings(_env_file=None, mock_mode=True)


# ---------------------------------------------------------------------------
# Prediction model: threshold + UNRELATED + edge conversion
# ---------------------------------------------------------------------------

def _pred(rel: RelationshipType, conf: float) -> RelationPrediction:
    return RelationPrediction(
        relationship_type=rel,
        confidence=conf,
        rationale="because I said so",
        detector="unit-test",
    )


def test_unrelated_is_never_storable():
    p = _pred(RelationshipType.UNRELATED, 0.99)
    assert p.is_storable is False
    assert p.accepts(0.0) is False            # even threshold 0 rejects it
    assert p.to_edge("iu_new", "iu_old", threshold=0.0) is None


def test_low_confidence_is_filtered_by_threshold():
    p = _pred(RelationshipType.SUPPORTS, 0.60)
    assert p.is_storable is True
    assert p.accepts(0.65) is False
    assert p.accepts(0.55) is True
    assert p.to_edge("iu_new", "iu_old", threshold=0.65) is None
    assert p.to_edge("iu_new", "iu_old", threshold=0.55) is not None


def test_to_edge_direction_follows_convention():
    """source --RELATIONSHIP--> target with rationale tagged by detector."""
    p = _pred(RelationshipType.CONTRADICTS, 0.8)
    edge = p.to_edge("iu_new", "iu_old", threshold=0.65)
    assert edge is not None
    assert edge.source_node_id == "iu_new"
    assert edge.target_node_id == "iu_old"
    assert edge.relationship_type is RelationshipType.CONTRADICTS
    assert edge.confidence == 0.8
    assert "[unit-test]" in edge.evidence_or_rationale
    assert "because I said so" in edge.evidence_or_rationale


def test_prediction_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        RelationPrediction(
            relationship_type=RelationshipType.SUPPORTS, confidence=1.4
        )


def test_as_label_is_human_readable():
    assert "SUPPORTS" in _pred(RelationshipType.SUPPORTS, 0.7).as_label()
    assert "0.70" in _pred(RelationshipType.SUPPORTS, 0.7).as_label()


# ---------------------------------------------------------------------------
# MockRelationDetector: all six types + determinism + errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,target,expected,min_conf", PAIRS)
def test_mock_detector_predicts_expected_type(detector, source, target, expected, min_conf):
    pred = detector.detect(source, target)
    assert pred.relationship_type is expected
    assert pred.confidence >= min_conf
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.rationale  # every prediction explains itself
    assert pred.detector == "mock-rules"


def test_mock_detector_is_deterministic(detector):
    a = detector.detect(PAIRS[1][0], PAIRS[1][1])
    b = detector.detect(PAIRS[1][0], PAIRS[1][1])
    assert a == b


def test_mock_detector_rejects_empty_texts(detector):
    with pytest.raises(RelationDetectorError, match="required"):
        detector.detect("", "something")
    with pytest.raises(RelationDetectorError, match="required"):
        detector.detect("something", "   ")


def test_weak_overlap_support_stays_below_default_threshold(detector):
    """Bare word sharing must NOT create an edge at the default threshold."""
    pred = detector.detect(
        "The committee met in March to discuss budget allocations.",
        "March is when the budget numbers were reviewed by auditors.",
    )
    assert pred.relationship_type is RelationshipType.SUPPORTS
    assert pred.confidence < 0.65
    assert pred.to_edge("iu_a", "iu_b", threshold=0.65) is None


def test_detect_units_uses_unit_content(detector):
    src = InformationUnit(
        content="Oil prices fell sharply on Tuesday.",
        information_type=InformationType.EVIDENCE,
        agent_id=AgentRole.RESEARCHER,
    )
    tgt = InformationUnit(
        content="Oil prices fell sharply on Tuesday.",
        information_type=InformationType.HYPOTHESIS,
        agent_id=AgentRole.PLANNER,
    )
    pred = detector.detect_units(src, tgt)
    assert pred.relationship_type is RelationshipType.REDUNDANT


def test_mock_pairs_produce_correct_edges_at_default_threshold(detector):
    """Integration: predictions -> edges, filtered by threshold + UNRELATED."""
    expected_edges = {
        (RelationshipType.SUPPORTS, 0.65),
        (RelationshipType.CONTRADICTS, 0.70),
        (RelationshipType.REFINES, 0.75),
        (RelationshipType.DEPENDS_ON, 0.70),
        (RelationshipType.REDUNDANT, 0.90),
        (RelationshipType.REDUNDANT, 0.80),
        # UNRELATED pair -> intentionally absent
    }
    got = set()
    for source, target, _expected_type, min_conf in PAIRS:
        pred = detector.detect(source, target)
        edge = pred.to_edge("iu_src", "iu_tgt", threshold=0.65)
        if edge is not None:
            got.add((edge.relationship_type, min_conf))
            assert edge.evidence_or_rationale  # rationale travels with the edge
    assert got == expected_edges


# ---------------------------------------------------------------------------
# LLMRelationDetector: schema, retries, errors
# ---------------------------------------------------------------------------

VALID_SUPPORT = json.dumps(
    {
        "relationship": "SUPPORTS",
        "confidence": 0.88,
        "rationale": "source data backs the target claim",
    }
)


@pytest.fixture
def llm_detector(settings):
    return LLMRelationDetector(MockLLMClient(), settings)


def test_llm_detector_parses_scripted_valid_reply(settings):
    llm = MockLLMClient(scripted={"RELATION_DETECTOR": VALID_SUPPORT})
    pred = LLMRelationDetector(llm, settings).detect("src text", "tgt text")
    assert pred.relationship_type is RelationshipType.SUPPORTS
    assert pred.confidence == 0.88
    assert pred.detector.startswith("llm:")
    # the pair actually reached the model
    assert "src text" in llm.calls[0]["user_text"]
    assert "tgt text" in llm.calls[0]["user_text"]


def test_llm_detector_rejects_invalid_relationship_value(settings):
    bad = json.dumps(
        {"relationship": "SUPPORT", "confidence": 0.9, "rationale": "typo"}
    )
    llm = MockLLMClient(scripted={"RELATION_DETECTOR": bad})
    with pytest.raises(RelationDetectorError, match="attempt"):
        LLMRelationDetector(llm, settings).detect("a", "b")
    assert len(llm.calls) == 3  # 1 try + 2 retries (default llm_json_retries=2)


def test_llm_detector_rejects_out_of_range_confidence(settings):
    bad = json.dumps(
        {"relationship": "SUPPORTS", "confidence": 7, "rationale": "oops"}
    )
    llm = MockLLMClient(scripted={"RELATION_DETECTOR": bad})
    with pytest.raises(RelationDetectorError):
        LLMRelationDetector(llm, settings).detect("a", "b")


def test_llm_detector_rejects_garbage_reply(settings):
    llm = MockLLMClient(scripted={"RELATION_DETECTOR": "I refuse to answer."})
    with pytest.raises(RelationDetectorError):
        LLMRelationDetector(llm, settings).detect("a", "b")


def test_llm_detector_wraps_connection_errors(settings):
    llm = MockLLMClient(fail_keys=["RELATION_DETECTOR"])
    with pytest.raises(RelationDetectorError, match="LLMConnectionError"):
        LLMRelationDetector(llm, settings).detect("a", "b")


def test_llm_detector_rejects_empty_inputs(llm_detector):
    with pytest.raises(RelationDetectorError, match="required"):
        llm_detector.detect("", "b")


def test_llm_detector_via_mock_llm_handles_real_pairs(llm_detector):
    """The canned RELATION_DETECTOR path in MockLLMClient stays schema-valid."""
    support = llm_detector.detect(PAIRS[0][0], PAIRS[0][1])
    assert support.relationship_type is RelationshipType.SUPPORTS
    contra = llm_detector.detect(PAIRS[1][0], PAIRS[1][1])
    assert contra.relationship_type is RelationshipType.CONTRADICTS


# ---------------------------------------------------------------------------
# factory + settings validation
# ---------------------------------------------------------------------------

def test_factory_uses_mock_detector_in_mock_mode(settings):
    detector = build_relation_detector(settings=settings)
    assert isinstance(detector, MockRelationDetector)


def test_factory_uses_llm_detector_when_mock_mode_off():
    s = Settings(_env_file=None, mock_mode=False, llm_api_key="k", llm_model="m")
    detector = build_relation_detector(llm=MockLLMClient(), settings=s)
    assert isinstance(detector, LLMRelationDetector)


def test_settings_reject_out_of_range_threshold():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, relation_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cosine_floor=-0.1)


# ---------------------------------------------------------------------------
# Optional NLI detector (injected fake model; real model skips if absent)
# ---------------------------------------------------------------------------

class _FakeNLI:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores]


def _nli(scores):
    from latent_workspace.relations.nli_detector import NLIRelationDetector

    return NLIRelationDetector("fake-nli", model=_FakeNLI(scores))


def test_nli_contradiction_mapping():
    pred = _nli([0.92, 0.05, 0.03]).detect(
        "The experiment failed to replicate the original finding.",
        "The experiment replicated the original finding.",
    )
    assert pred.relationship_type is RelationshipType.CONTRADICTS
    assert pred.confidence == pytest.approx(0.92, abs=0.01)
    assert "contradiction" in pred.rationale


def test_nli_entailment_maps_to_supports():
    pred = _nli([0.02, 0.08, 0.90]).detect(
        "Regional revenues grew 4% last quarter.",
        "Revenues increased during the quarter.",
    )
    assert pred.relationship_type is RelationshipType.SUPPORTS
    assert pred.confidence == pytest.approx(0.90, abs=0.01)


def test_nli_neutral_maps_to_unrelated():
    pred = _nli([0.05, 0.88, 0.07]).detect(
        "The football team won the final.",
        "Telescope images show a distant galaxy cluster.",
    )
    assert pred.relationship_type is RelationshipType.UNRELATED


def test_nli_lexical_override_redundant():
    same = "Oil prices fell sharply on Tuesday."
    pred = _nli([0.5, 0.4, 0.1]).detect(same, same)  # scores would say contradiction
    assert pred.relationship_type is RelationshipType.REDUNDANT


def test_nli_lexical_override_depends_on():
    pred = _nli([0.05, 0.5, 0.45]).detect(
        "This forecast depends on the assumption of stable demand.",
        "Demand stayed stable this year.",
    )
    assert pred.relationship_type is RelationshipType.DEPENDS_ON


def test_nli_real_model_end_to_end():
    from latent_workspace.relations.nli_detector import NLIRelationDetector

    try:
        detector = NLIRelationDetector()
    except RelationDetectorError as exc:
        pytest.skip(f"real NLI model unavailable: {exc}")
    pred = detector.detect(
        "Vaccines do not prevent the disease in this cohort.",
        "Vaccines prevent the disease.",
    )
    assert pred.relationship_type in (
        RelationshipType.CONTRADICTS,
        RelationshipType.SUPPORTS,
        RelationshipType.UNRELATED,
        RelationshipType.REDUNDANT,
    )
    assert 0.0 <= pred.confidence <= 1.0


def test_relations_package_lazy_exports_nli():
    from latent_workspace import relations

    assert relations.NLIRelationDetector is not None
    assert "NLIRelationDetector" in relations.__all__
