"""Chunk 7 demo: plan -> workers -> graph -> grounded final answer.

    python scripts/demo_orchestrator.py

Shows, fully offline:
 1. a workspace seeded with evidence, counter-evidence and noise
 2. the Planner's subtasks and hypotheses entering the shared graph
 3. each worker receiving a retrieved subgraph (not raw chat history)
 4. every new unit and predicted edge being ingested
 5. the Synthesizer's cited final answer, plus run metrics/errors

Reminder: the graph stores PREDICTED relationships with confidence; it
does not prove that any statement is factually correct.
"""

from __future__ import annotations

import sys

from latent_workspace.config import get_settings
from latent_workspace.graph import build_workspace
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
    SubtaskStatus,
)
from latent_workspace.orchestrator import build_orchestrator, extract_citations

QUERY = "Why did global chip prices rise in 2025?"

SEED_UNITS = [
    (
        "E1 evidence",
        "Global chip prices rose 18% in 2025, driven by AI datacenter demand.",
        InformationType.EVIDENCE,
        AgentRole.RESEARCHER,
        0.82,
    ),
    (
        "C1 counter-evidence",
        "Some analysts argue chip prices did not rise in 2025 after inflation.",
        InformationType.COUNTER_EVIDENCE,
        AgentRole.CRITIC,
        0.72,
    ),
    (
        "U1 unrelated",
        "The football team won the championship final.",
        InformationType.FACT,
        AgentRole.RESEARCHER,
        0.70,
    ),
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
    labels: dict[str, str] = {}
    for label, content, u_type, agent, conf in SEED_UNITS:
        unit = InformationUnit(
            content=content,
            information_type=u_type,
            agent_id=agent,
            confidence=conf,
        )
        graph.add_unit(unit)
        labels[unit.id] = label

    line("1) SEEDED WORKSPACE")
    print(f"query: {QUERY!r}")
    print(f"nodes={graph.node_count} edges={graph.edge_count}")
    for unit in graph.nodes():
        print(f"  {labels.get(unit.id, unit.id)}: {unit.content[:66]}")

    result = build_orchestrator(settings, graph=graph).run(QUERY)

    line("2) PLAN AND SUBTASK OUTCOMES")
    print(
        f"rounds={result.metrics.rounds} agent_calls={result.metrics.agent_calls} "
        f"nodes_added={result.metrics.nodes_added} "
        f"edges_added={result.metrics.edges_added}"
    )
    for subtask in result.subtasks:
        marker = "done" if subtask.status is SubtaskStatus.DONE else "not done"
        print(
            f"  [{marker:8s}] {subtask.assigned_agent.value:10s} "
            f"{subtask.id} units={len(subtask.unit_ids)} :: "
            f"{subtask.description[:58]}"
        )
        if subtask.notes:
            print(f"             note: {subtask.notes}")

    line("3) CONTEXTS HANDED TO AGENTS")
    for index, context in enumerate(result.contexts, start=1):
        print(
            f"\n  context {index}: {context.task_description[:64]!r} "
            f"items={len(context)} tokens~{context.token_estimate} "
            f"truncated={context.truncated}"
        )
        for item in context.items:
            name = labels.get(item.unit.id, item.unit.id)
            print(
                f"    {name:18s} [{item.stance:12s}] "
                f"{item.unit.content[:62]}"
            )
        if context.contradicting_ids:
            names = [labels.get(uid, uid) for uid in context.contradicting_ids]
            print(f"    contradictions surfaced: {names}")

    line("4) NEW PREDICTED EDGES")
    if not result.edges:
        print("  (none this run)")
    for edge in result.edges:
        source = labels.get(edge.source_node_id, edge.source_node_id)
        target = labels.get(edge.target_node_id, edge.target_node_id)
        print(
            f"  {source} --{edge.relationship_type.value}--> {target} "
            f"(conf={edge.confidence:.2f})"
        )
        if edge.evidence_or_rationale:
            print(f"    rationale: {edge.evidence_or_rationale[:100]}")

    line("5) FINAL ANSWER")
    print(result.final_answer)
    citations = extract_citations(result.final_answer)
    print(
        f"\ncitations={citations or 'none'} "
        f"grounded={result.extra.get('answer_grounded')} "
        f"mode={result.extra.get('mode')}"
    )
    if result.errors:
        print("\nrecorded errors (run continued):")
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("\nrecorded errors: none")

    line("NOTE")
    print("Edges above are PREDICTED relationships with confidence scores.")
    print("The final answer cites workspace units; it is not verified truth.")
    print("OK - chunk 7 demo finished with zero API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
