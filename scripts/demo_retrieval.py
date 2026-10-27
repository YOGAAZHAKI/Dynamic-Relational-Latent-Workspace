"""Chunk 6 demo: query -> retrieved, scored, stance-labeled subgraph.

    python scripts/demo_retrieval.py

Shows, fully offline:
 1. a small workspace (units -> typed edges, built in chunk 5)
 2. retrieval scores: semantic match + relational signal + hop
 3. the retrieved subgraph: stances (SUPPORTING / CONTRADICTING),
    typed confidence-labeled relations, provenance chains
 4. what was left out: irrelevant nodes vs budget exclusions
 5. the exact context block handed to a downstream agent (and the
    agent actually receiving it via the offline mock LLM)
 6. a tiny-budget pass proving conflicts are flagged, not silently cut

Reminder: edges are PREDICTIONS with confidence - the graph does not
prove any statement factually correct.
"""

from __future__ import annotations

import sys

from latent_workspace.agents import ResearcherAgent
from latent_workspace.config import get_settings
from latent_workspace.graph import build_workspace
from latent_workspace.llm import MockLLMClient
from latent_workspace.models import (
    AgentRole,
    InformationType,
    InformationUnit,
)
from latent_workspace.retrieval import SubgraphRetriever

UNITS = [
    ("H1 hypothesis", "Electricity prices rose for industrial users.",
     InformationType.HYPOTHESIS, AgentRole.PLANNER, 0.60),
    ("E1 evidence", "Regional data confirms that industrial electricity prices increased sharply.",
     InformationType.EVIDENCE, AgentRole.RESEARCHER, 0.80),
    ("C1 counter-evidence", "Industrial electricity prices do not rise for industrial users.",
     InformationType.COUNTER_EVIDENCE, AgentRole.CRITIC, 0.70),
    ("R1 refinement", "Industrial electricity prices rose approximately 12% for industrial users in Q3.",
     InformationType.REFINEMENT, AgentRole.CRITIC, 0.75),
    ("F1 unrelated", "The football team won the championship final.",
     InformationType.FACT, AgentRole.RESEARCHER, 0.70),
]

QUERY = "did industrial electricity prices rise for industrial users?"


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    settings = get_settings()
    if not settings.mock_mode:
        print("NOTE: DLRW_MOCK_MODE=false -> this demo would call a real LLM.")
        print("      Set DLRW_MOCK_MODE=true for a free offline demo.\n")

    # -- 1. workspace ----------------------------------------------------
    graph = build_workspace(settings)
    by_label: dict[str, InformationUnit] = {}
    for label, content, u_type, agent, conf in UNITS:
        unit = InformationUnit(
            content=content, information_type=u_type, agent_id=agent,
            confidence=conf,
        )
        graph.add_unit(unit)
        by_label[label] = unit
    labels = {u.id: label for label, u in by_label.items()}

    line("1) WORKSPACE (from chunk 5)")
    print(f"nodes: {graph.node_count} | edges: {graph.edge_count}")
    for edge in graph.edges():
        print(f"  {labels.get(edge.source_node_id, edge.source_node_id)}"
              f" --{edge.relationship_type.value}--> "
              f"{labels.get(edge.target_node_id, edge.target_node_id)}"
              f"  (conf={edge.confidence:.2f})")

    retriever = SubgraphRetriever(graph)

    # -- 2. ranking ------------------------------------------------------
    line(f"2) RETRIEVAL SCORES for: {QUERY!r}")
    print("score = 0.6 * cosine(query, unit) + 0.4 * edge-signal (decays per hop)")
    for entry in retriever.rank(QUERY):
        name = labels.get(entry.node_id, entry.node_id)
        print(f"  {name:16s} score={entry.score:.3f} "
              f"(semantic={entry.semantic:.3f}, relational={entry.relational:.3f}, "
              f"hop={entry.hop})")

    # -- 3. the retrieved subgraph ---------------------------------------
    ctx = retriever.retrieve(QUERY)
    line("3) RETRIEVED SUBGRAPH (context for the next agent)")
    print(f"anchor (query center): {labels.get(ctx.anchor_id, ctx.anchor_id)}")
    print(f"supporting:  {[labels.get(i, i) for i in ctx.supporting_ids]}")
    print(f"contradicting: {[labels.get(i, i) for i in ctx.contradicting_ids]}")
    for item in ctx.items:
        name = labels.get(item.unit.id, item.unit.id)
        print(f"\n  {name}  [{item.stance.upper()}]")
        print(f"    {item.unit.content[:74]}")
        for label in item.relation_labels:
            print(f"    relation: {label}")
        for p in item.provenance:
            src = labels.get(p.unit_id, p.unit_id)
            print(f"    provenance: {src} --{p.relationship_type.value}--> "
                  f"{name} (conf={p.confidence:.2f}, depth={p.depth})")

    # -- 4. what was left out --------------------------------------------
    line("4) WHAT WAS LEFT OUT")
    reached_ids = set(ctx.unit_ids()) | set(ctx.excluded_unit_ids)
    for label, unit in by_label.items():
        if unit.id in reached_ids:
            continue
        print(f"  {label}: never retrieved (irrelevant to the query: "
              "no semantic match, no walkable edges)")
    if ctx.excluded_unit_ids:
        print("  budget-excluded (reached but not shown):")
        for nid in ctx.excluded_unit_ids:
            print(f"    {labels.get(nid, nid)}")
    else:
        print("  budget-excluded: none (everything reached fit the budget)")

    # -- 5. the context handed to an agent --------------------------------
    line("5) CONTEXT PASSED TO A DOWNSTREAM AGENT")
    print(ctx.as_prompt_block())
    print(f"\n(token_estimate={ctx.token_estimate}, truncated={ctx.truncated})")

    llm = MockLLMClient()
    agent = ResearcherAgent(llm, settings)
    result = agent.run("Summarize the evidence on industrial electricity prices", ctx)
    user_text = llm.calls[0]["user_text"]
    print(f"\nresearcher agent ran offline: error={result.error!r}, "
          f"units produced={len(result.units)}")
    print(f"context reached the model prompt: "
          f"{'yes' if ctx.items[0].unit.id in user_text else 'NO'}; "
          f"conflict labels present: "
          f"{'yes' if '[CONTRADICTING]' in user_text else 'NO'}")

    # -- 6. tiny budget: conflicts flagged, not silently dropped ----------
    line("6) TINY TOKEN BUDGET (conflicts must stay visible)")
    tiny = retriever.retrieve(QUERY, token_budget=80)
    print(f"items kept: {[labels.get(i, i) for i in tiny.unit_ids()]}")
    print(f"excluded (flagged, not hidden): "
          f"{[labels.get(i, i) for i in tiny.excluded_unit_ids]}")
    print(f"truncated={tiny.truncated} -> agents are told the list is partial")

    line("NOTE")
    print("Edges and provenance above are PREDICTED relationships with")
    print("confidence scores - not proof of factual correctness.")
    print("OK - chunk 6 demo finished with zero API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
