"""Chunk 2 demo: run all four agents OFFLINE (no API key, no cost).

    python scripts/demo_mock.py

Shows: planner -> researcher -> critic -> synthesizer, with structured
outputs, token counts and error handling. Later chunks add the graph.
"""

from __future__ import annotations

import sys

from latent_workspace.agents import build_agents
from latent_workspace.config import get_settings
from latent_workspace.llm import build_llm
from latent_workspace.models import ReasoningContext

QUERY = "Why did global chip prices rise in 2025, and will the trend continue?"


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    settings = get_settings()
    if not settings.mock_mode:
        print("NOTE: DLRW_MOCK_MODE is false -> this demo would call a real LLM.")
        print("      Set DLRW_MOCK_MODE=true for a free offline demo.\n")

    llm = build_llm(settings)
    agents = build_agents(llm, settings)
    print(f"LLM client: {llm.model_name} | mock_mode={settings.mock_mode}")

    # -- 1. Planner --------------------------------------------------------
    line("1) PLANNER")
    plan = agents.planner.run(QUERY)
    if not plan.succeeded:
        print(f"FAILED: {plan.error}")
        return 1
    print(f"Subtasks ({len(plan.subtasks)}):")
    for i, st in enumerate(plan.subtasks, 1):
        print(f"  {i}. [{st.assigned_agent.value}] {st.description}  ({st.id})")
    print(f"Hypotheses -> {len(plan.units)} unit(s):")
    for u in plan.units:
        print(f"  {u.short()}")

    # -- 2. Researcher (with an empty context for now) ---------------------
    line("2) RESEARCHER  (first subtask, empty workspace context)")
    subtask = plan.subtasks[0]
    research = agents.researcher.run(subtask.description, None, subtask_id=subtask.id)
    if not research.succeeded:
        print(f"FAILED: {research.error}")
        return 1
    for u in research.units:
        print(f"  {u.short()}")
        print(f"      sources={u.metadata['sources']} depends_on={u.metadata['depends_on']}")

    # -- 3. Critic (given the planner's hypotheses as context) -------------
    line("3) CRITIC  (hypotheses supplied as workspace context)")
    hyp_ctx = ReasoningContext(
        task_description="Audit the hypotheses",
        items=[],  # Chunk 5/6 fill this from the graph; empty is legal today
    )

    critic = agents.critic.run(
        "Audit the hypotheses and evidence for the price-rise question",
        hyp_ctx,
        subtask_id=subtask.id,
    )
    if not critic.succeeded:
        print(f"FAILED: {critic.error}")
        return 1
    for u in critic.units:
        print(f"  {u.short()}")
        print(f"      target={u.metadata.get('target')} issues={u.metadata['issues']}")

    # -- 4. Synthesizer ----------------------------------------------------
    line("4) SYNTHESIZER  (mock context)")
    synth = agents.synthesizer.run(QUERY, None)
    if not synth.succeeded:
        print(f"FAILED: {synth.error}")
        return 1
    print(synth.answer)

    # -- summary -----------------------------------------------------------
    line("SUMMARY")
    results = [plan, research, critic, synth]
    total_tokens = sum(r.tokens_used for r in results)
    total_ms = sum(r.duration_ms for r in results)
    total_units = sum(len(r.units) for r in results)
    print(f"agent calls      : {len(results)}")
    print(f"information units: {total_units}")
    print(f"subtasks         : {len(plan.subtasks)}")
    print(f"tokens used      : {total_tokens} (estimated in mock mode)")
    print(f"latency          : {total_ms:.1f} ms")
    print("OK - chunk 2 demo finished with zero API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
