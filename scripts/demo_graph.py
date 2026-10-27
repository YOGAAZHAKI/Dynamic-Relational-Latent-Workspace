"""Chunk 5 demo: information units becoming a relational graph.

    python scripts/demo_graph.py

Shows, fully offline:
 1. units entering the workspace one by one (embed -> candidates ->
    classify -> threshold -> commit)
 2. predicted relationships becoming labeled, confidence-scored edges
 3. queries: supporting evidence, contradicting evidence, provenance
 4. snapshot save + load round trip

Reminder: edges are PREDICTIONS with confidence - the graph does not
prove any statement factually correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

from latent_workspace.config import get_settings
from latent_workspace.embeddings import MockEmbedder
from latent_workspace.graph import build_workspace
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
)
from latent_workspace.relations import MockRelationDetector

UNITS = [
    # (label, content, type, agent, confidence)
    ("H1 hypothesis", "Electricity prices rose for industrial users.",
     InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.60),
    ("E1 evidence", "Regional data confirms that industrial electricity prices increased sharply.",
     InformationType.EVIDENCE, AgentRole.RESEARCHER, 0.80),
    ("C1 counter-evidence", "Industrial electricity prices do not rise for industrial users.",
     InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.70),
    ("R1 refinement", "Industrial electricity prices rose approximately 12% for industrial users in Q3.",
     InformationType.REFINEMENT, AgentRole.CRITIC, 0.75),
    ("U1 unrelated", "The football team won the championship final.",
     InformationType.FACT, AgentRole.RESEARCHER, 0.70),
    ("W1 weak overlap", "Industrial electricity sector report was published in March.",
     InformationType.FACT, AgentRole.RESEARCHER, 0.70),
]


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    settings = get_settings()
    if not settings.mock_mode:
        print("NOTE: DLRW_MOCK_MODE=false -> this demo would call a real LLM.")
        print("      Set DLRW_MOCK_MODE=true for a free offline demo.\n")

    graph = build_workspace(settings)
    print(
        f"components: embedder={graph.embedder.model_name} "
        f"| store=FAISS({graph.store.dim}) "
        f"| detector={graph.detector.name}"
    )

    line("1) ADDING INFORMATION UNITS (incremental workspace updates)")
    units: dict[str, InformationUnit] = {}
    for label, content, u_type, agent, conf in UNITS:
        unit = InformationUnit(
            content=content, information_type=u_type,
            agent_id=agent, confidence=conf,
        )
        units[label] = unit
        result = graph.add_unit(unit)
        print(f"\n  {label}: {unit.content[:70]}")
        print(f"      node {result.unit_id} | candidates={len(result.candidate_ids)}"
              f" | edges added={result.added_edge_count}")
        for edge in result.edges_added:
            print(f"      + {edge.direction_text()}")
            print(f"        rationale: {edge.evidence_or_rationale}")
        for reason in result.skipped:
            print(f"      - skipped: {reason}")
        for err in result.errors:
            print(f"      ! error: {err}")

    line("2) GRAPH AT A GLANCE")
    print(f"nodes: {graph.node_count} | edges: {graph.edge_count}")
    by_type: dict[str, int] = {}
    for edge in graph.edges():
        by_type[edge.relationship_type.value] = by_type.get(edge.relationship_type.value, 0) + 1
    print(f"edge types: {by_type}")

    h = units["H1 hypothesis"]
    line("3) QUERIES ON THE HYPOTHESIS NODE")
    print(f"node: {h.id}  \"{h.content}\"")
    print("\nSUPPORTING evidence:")
    for unit, edge in graph.find_supporting(h.id):
        print(f"  <- {unit.id} (conf={edge.confidence:.2f}) {unit.content[:60]}")
    print("\nCONTRADICTING evidence:")
    for unit, edge in graph.find_contradicting(h.id):
        print(f"  <- {unit.id} (conf={edge.confidence:.2f}) {unit.content[:60]}")
    print("\nPROVENANCE chain (SUPPORTS / DEPENDS_ON / REFINES only):")
    steps = graph.provenance(h.id)
    if not steps:
        print("  (no grounding chain yet)")
    for step in steps:
        indent = "  " * step.depth
        print(f"  {indent}depth {step.depth}: {step.unit.id} "
              f"--{step.edge.relationship_type.value}--> {h.id}")

    line("4) SAVE / LOAD SNAPSHOT")
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    snapshot = graph.save(out_dir / "demo_graph.json")
    print(f"saved: {snapshot} (+ {snapshot.with_suffix('.faiss').name})")

    restored = type(graph).load(
        snapshot,
        embedder=MockEmbedder(graph.embedder.dim),
        detector=MockRelationDetector(),
        settings=settings,
    )
    print(f"restored: nodes={restored.node_count}, edges={restored.edge_count}, "
          f"vector ids={restored.store.size}")
    same = (
        {u.id for u in restored.nodes()} == {u.id for u in graph.nodes()}
        and {e.id for e in restored.edges()} == {e.id for e in graph.edges()}
    )
    print(f"identical to original: {same}")

    line("NOTE")
    print("Edges above are PREDICTED relationships with confidence scores.")
    print("The graph does NOT prove any statement factually correct.")
    print("OK - chunk 5 demo finished with zero API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
