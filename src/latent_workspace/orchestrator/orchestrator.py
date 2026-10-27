"""End-to-end multi-agent orchestration (chunk 7).

The orchestrator is deliberately thin glue: it never re-implements an
agent, a graph query or a retrieval policy. It only decides *what runs
next* and keeps the shared workspace fed:

    query
      -> Planner            (subtasks + hypotheses)
      -> per subtask:       retrieve subgraph -> Researcher / Critic
      -> every unit is ingested into the KnowledgeGraph
      -> Synthesizer        (grounded final answer)

Design rules
------------
- **Crash-proof**: an agent or retrieval failure becomes an entry in
  ``OrchestratorResult.errors``; the run continues whenever possible.
- **Grounded**: the Synthesizer only sees a retrieved ``ReasoningContext``.
  If synthesis fails, a deterministic extractive fallback still cites
  real unit ids instead of inventing an answer.
- **Auditable**: every context handed to an agent is kept in
  ``OrchestratorResult.contexts`` and every added edge in ``.edges``.
- **Offline first**: with ``DLRW_MOCK_MODE=true`` a full run costs
  nothing and produces identical output run-to-run.
"""

from __future__ import annotations

import re
import time

from ..agents import AgentSet, build_agents
from ..config import Settings, get_settings
from ..graph import KnowledgeGraph, build_workspace
from ..llm import LLMClient, build_llm
from ..models import (
    AgentResult,
    AgentRole,
    InformationUnit,
    OrchestratorResult,
    ReasoningContext,
    RunMetrics,
    Subtask,
    SubtaskStatus,
)
from ..retrieval import SubgraphRetriever, build_retriever

_CITATION_RE = re.compile(r"\[?(iu_[0-9a-fA-F]{8})\]?")

#: worker roles the orchestrator knows how to dispatch
_WORKERS = frozenset({AgentRole.RESEARCHER, AgentRole.CRITIC})


class OrchestratorError(Exception):
    """Invalid orchestration input or an unrecoverable setup failure."""


def extract_citations(text: str) -> list[str]:
    """Return unique unit ids cited in ``text``, preserving order."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def fallback_answer(query: str, context: ReasoningContext) -> str:
    """Deterministic extractive answer used only when synthesis fails.

    It quotes retrieved units (with citations) and surfaces conflicts; it
    does not add facts of its own.
    """
    if not context.items:
        return (
            "I could not find enough evidence in the workspace to answer "
            f"the query: {query}"
        )
    lines = [
        "Synthesizer unavailable; extractive fallback from the retrieved "
        "workspace (predicted relations, not verified truth):"
    ]
    for item in context.items[:5]:
        label = item.unit.information_type.value
        stance = f", {item.stance}" if item.stance != "neutral" else ""
        lines.append(
            f"- [{item.unit.id}] ({label}, conf={item.unit.confidence:.2f}"
            f"{stance}) {item.unit.content}"
        )
    if len(context.items) > 5:
        lines.append(f"- ... and {len(context.items) - 5} more retrieved unit(s).")
    if context.contradicting_ids:
        lines.append(
            "Note: the retrieved evidence contains conflicting units "
            f"({', '.join(context.contradicting_ids)}); this disagreement "
            "is not resolved by the fallback."
        )
    return "\n".join(lines)


class Orchestrator:
    """Coordinates Planner -> workers -> Synthesizer over one workspace."""

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        agents: AgentSet | None = None,
        retriever: SubgraphRetriever | None = None,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        if llm is None and graph is not None:
            llm = getattr(getattr(graph, "detector", None), "llm", None)
        self.llm = llm

        self.graph = graph if graph is not None else build_workspace(
            self.settings, llm=llm
        )
        if agents is None:
            agent_llm = llm if llm is not None else build_llm(self.settings)
            agents = build_agents(agent_llm, self.settings)
        self.agents = agents
        if retriever is None:
            # A run's own generated units compete with the source units
            # for seed slots. Orchestration therefore widens the semantic
            # seed row (2x top_k) so relevant evidence is not crowded out
            # before graph expansion can reach counter-evidence.
            seed_k = self.settings.top_k * 2
            retriever = build_retriever(
                self.graph, self.settings, seed_k=seed_k
            )
        self.retriever = retriever

        # one shared client keeps token accounting meaningful; when only
        # one side was injected we fall back to the agents' client.
        if self.llm is None:
            self.llm = agents.planner.llm

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def run(self, query: str, *, max_rounds: int | None = None) -> OrchestratorResult:
        """Execute one full reasoning run and return an auditable result."""
        task = self._require_query(query)
        started = time.perf_counter()

        metrics = RunMetrics()
        errors: list[str] = []
        subtasks: list[Subtask] = []
        units: list[InformationUnit] = []
        edges = []
        contexts: list[ReasoningContext] = []

        # -- 1. planning ---------------------------------------------------
        plan = self._call_agent(self.agents.planner, task, None, None, metrics)
        if plan.succeeded:
            subtasks = list(plan.subtasks)
            self._ensure_worker_coverage(subtasks, task)
            self._ingest(plan.units, units, edges, metrics, errors, "PLANNER")
        else:
            errors.append(f"PLANNER: {plan.error}")
            subtasks = [
                Subtask(
                    description=f"Gather evidence and counter-evidence to answer: {task}",
                    assigned_agent=AgentRole.RESEARCHER,
                )
            ]

        # -- 2. worker rounds ----------------------------------------------
        cap = max(0, int(self.settings.max_rounds if max_rounds is None else max_rounds))
        rounds = 0
        while rounds < cap:
            pending = [st for st in subtasks if st.status is SubtaskStatus.PENDING]
            if not pending:
                break
            rounds += 1
            for subtask in pending:
                self._run_subtask(
                    subtask, units, edges, contexts, metrics, errors
                )

        metrics.rounds = rounds

        # -- 3. synthesis ---------------------------------------------------
        final_ctx = self._retrieve(task, errors, phase="SYNTHESIZER")
        contexts.append(final_ctx)
        synthesis = self._call_agent(
            self.agents.synthesizer, task, final_ctx, None, metrics
        )
        if synthesis.succeeded and synthesis.answer.strip():
            answer = synthesis.answer.strip()
        else:
            if synthesis.error:
                errors.append(f"SYNTHESIZER: {synthesis.error}")
            answer = fallback_answer(task, final_ctx)

        citations = extract_citations(answer)
        valid_citations = [cid for cid in citations if self.graph.has_node(cid)]

        metrics.latency_ms = (time.perf_counter() - started) * 1000.0
        return OrchestratorResult(
            query=task,
            final_answer=answer,
            subtasks=subtasks,
            units=units,
            edges=edges,
            contexts=contexts,
            metrics=metrics,
            errors=errors,
            extra={
                "mode": "mock" if self.settings.is_mock else "live",
                "graph_nodes": self.graph.node_count,
                "graph_edges": self.graph.edge_count,
                "answer_citations": valid_citations,
                "answer_grounded": (
                    bool(citations) and len(valid_citations) == len(citations)
                ),
                "planner_failed": not plan.succeeded,
                "synthesizer_failed": not synthesis.succeeded,
                "rounds_requested": cap,
            },
        )

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def _run_subtask(
        self,
        subtask: Subtask,
        units: list[InformationUnit],
        edges,
        contexts: list[ReasoningContext],
        metrics: RunMetrics,
        errors: list[str],
    ) -> None:
        if subtask.assigned_agent not in _WORKERS:
            subtask.status = SubtaskStatus.FAILED
            subtask.notes = f"unsupported worker role {subtask.assigned_agent.value}"
            errors.append(f"SUBTASK {subtask.id}: {subtask.notes}")
            return

        subtask.status = SubtaskStatus.IN_PROGRESS
        context = self._retrieve(
            subtask.description, errors, phase=f"SUBTASK {subtask.id}"
        )
        contexts.append(context)

        worker = (
            self.agents.researcher
            if subtask.assigned_agent is AgentRole.RESEARCHER
            else self.agents.critic
        )
        result = self._call_agent(
            worker, subtask.description, context, subtask.id, metrics
        )
        if not result.succeeded:
            subtask.status = SubtaskStatus.FAILED
            subtask.notes = result.error or "agent failed"
            errors.append(f"{subtask.assigned_agent.value} {subtask.id}: {result.error}")
            return

        before = len(units)
        self._ingest(
            result.units,
            units,
            edges,
            metrics,
            errors,
            f"{subtask.assigned_agent.value} {subtask.id}",
        )
        subtask.unit_ids = [u.id for u in units[before:]]
        subtask.status = SubtaskStatus.DONE

    def _retrieve(
        self,
        query: str,
        errors: list[str],
        *,
        phase: str,
    ) -> ReasoningContext:
        budget = min(
            self.settings.retrieval_token_budget,
            self.settings.max_context_tokens,
        )
        try:
            return self.retriever.retrieve(query, token_budget=budget)
        except Exception as exc:  # noqa: BLE001 - retrieval must not kill a run
            errors.append(f"{phase}: retrieval failed ({type(exc).__name__}): {exc}")
            return ReasoningContext(task_description=query)

    def _call_agent(
        self,
        agent,
        task: str,
        context: ReasoningContext | None,
        subtask_id: str | None,
        metrics: RunMetrics,
    ) -> AgentResult:
        result = agent.run(task, context, subtask_id=subtask_id)
        metrics.agent_calls += 1
        metrics.total_tokens += int(result.tokens_used or 0)
        return result

    def _ingest(
        self,
        incoming: list[InformationUnit],
        units: list[InformationUnit],
        edges,
        metrics: RunMetrics,
        errors: list[str],
        phase: str,
    ) -> None:
        """Add units to the shared graph; never let one bad unit abort a run."""
        for unit in incoming:
            if self.graph.has_node(unit.id):
                units.append(unit)
                continue
            try:
                added = self.graph.add_unit(unit)
            except Exception as exc:  # noqa: BLE001 - graph failure is per-unit
                errors.append(
                    f"{phase}: failed to add {unit.id} "
                    f"({type(exc).__name__}): {exc}"
                )
                continue
            units.append(unit)
            edges.extend(added.edges_added)
            metrics.nodes_added += 1
            metrics.edges_added += added.added_edge_count
            for detail in added.errors:
                errors.append(f"{phase}: {detail}")

    def _ensure_worker_coverage(
        self, subtasks: list[Subtask], query: str
    ) -> None:
        """The system always researches and always critiques.

        A Planner that omits one side would silently weaken the pipeline,
        so missing coverage is added deterministically here.
        """
        roles = {st.assigned_agent for st in subtasks}
        if AgentRole.RESEARCHER not in roles:
            subtasks.append(
                Subtask(
                    description=f"Gather key evidence needed to answer: {query}",
                    assigned_agent=AgentRole.RESEARCHER,
                )
            )
        if AgentRole.CRITIC not in roles:
            subtasks.append(
                Subtask(
                    description=(
                        "Scrutinize the hypotheses and evidence for weaknesses "
                        f"or counter-evidence about: {query}"
                    ),
                    assigned_agent=AgentRole.CRITIC,
                )
            )

    @staticmethod
    def _require_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise OrchestratorError("query must be a non-blank string")
        return query.strip()


def build_orchestrator(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    graph: KnowledgeGraph | None = None,
    agents: AgentSet | None = None,
    retriever: SubgraphRetriever | None = None,
) -> Orchestrator:
    """Factory: a fully wired orchestrator for the current settings."""
    return Orchestrator(
        graph=graph,
        agents=agents,
        retriever=retriever,
        settings=settings,
        llm=llm,
    )
