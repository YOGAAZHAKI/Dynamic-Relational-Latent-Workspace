"""Systems compared by the evaluation harness (chunk 9).

A = LLM only, no retrieval
B = flat context (every workspace unit, no relations)
C = vector RAG (semantic seeds only, no graph expansion)
D = full DRLW (typed graph + subgraph retrieval + orchestration)
Ablation = DRLW without relational expansion (depth = 0)

Every system returns the same :class:`EvalRun` record, so metrics and
reports never special-case one implementation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, Field

from ..agents import SynthesizerAgent
from ..config import Settings, get_settings
from ..graph import KnowledgeGraph, build_workspace
from ..llm import LLMClient, build_llm
from ..models import ContextItem, InformationUnit, ReasoningContext
from ..orchestrator import Orchestrator, build_orchestrator
from ..retrieval import SubgraphRetriever
from ..retrieval.base import estimate_item_tokens
from .dataset import EvalCase

LLMFactory = Callable[[], LLMClient]


class EvalRun(BaseModel):
    """Everything one system produced for one case (auditable record)."""

    system: str
    case_id: str
    query: str
    answer: str = ""
    retrieved_unit_ids: list[str] = Field(default_factory=list)
    retrieved_contents: list[str] = Field(default_factory=list)
    workspace_contents: dict[str, str] = Field(default_factory=dict)
    context_text: str = ""
    context_tokens: int = 0
    total_tokens: int = 0
    agent_calls: int = 0
    nodes: int = 0
    edges: int = 0
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class EvalSystem(ABC):
    """Base class: build a fresh workspace/LLM for every case."""

    name: str
    description: str

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm_factory = llm_factory

    # -- abstract ---------------------------------------------------------
    @abstractmethod
    def run_case(self, case: EvalCase) -> EvalRun: ...

    # -- helpers ----------------------------------------------------------
    def _new_llm(self) -> LLMClient:
        if self._llm_factory is not None:
            return self._llm_factory()
        return build_llm(self.settings)

    def _new_graph(self, llm: LLMClient) -> KnowledgeGraph:
        return build_workspace(self.settings, llm=llm)

    def _populate(
        self, graph: KnowledgeGraph, case: EvalCase
    ) -> tuple[dict[str, str], list[str]]:
        """Insert the case's units; return (id -> content, errors)."""
        contents: dict[str, str] = {}
        errors: list[str] = []
        for spec in case.units:
            unit = InformationUnit(
                content=spec.content,
                information_type=spec.information_type,
                agent_id=spec.agent_id,
                confidence=spec.confidence,
                metadata={"case_id": case.id, "evaluation": True},
            )
            try:
                graph.add_unit(unit)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                errors.append(
                    f"add_unit {unit.id} failed ({type(exc).__name__}): {exc}"
                )
                continue
            contents[unit.id] = unit.content
        return contents, errors

    @staticmethod
    def _context_items(units: list[InformationUnit]) -> list[ContextItem]:
        return [ContextItem(unit=unit) for unit in units]

    def _finalize(
        self,
        case: EvalCase,
        *,
        answer: str,
        graph: KnowledgeGraph | None,
        contexts: list[ReasoningContext],
        total_tokens: int,
        agent_calls: int,
        started: float,
        errors: list[str],
        extra: dict | None = None,
    ) -> EvalRun:
        workspace_contents = (
            {unit.id: unit.content for unit in graph.nodes()} if graph else {}
        )
        retrieved_ids: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            for item in context.items:
                if item.unit.id not in seen:
                    seen.add(item.unit.id)
                    retrieved_ids.append(item.unit.id)
        retrieved_contents = [
            workspace_contents[i] for i in retrieved_ids if i in workspace_contents
        ]
        context_text = "\n\n".join(c.as_prompt_block() for c in contexts)
        context_tokens = sum(c.token_estimate for c in contexts)
        if context_tokens == 0 and context_text:
            context_tokens = max(1, len(context_text) // 4)

        return EvalRun(
            system=self.name,
            case_id=case.id,
            query=case.query,
            answer=(answer or "").strip(),
            retrieved_unit_ids=retrieved_ids,
            retrieved_contents=retrieved_contents,
            workspace_contents=workspace_contents,
            context_text=context_text,
            context_tokens=context_tokens,
            total_tokens=total_tokens,
            agent_calls=agent_calls,
            nodes=graph.node_count if graph else 0,
            edges=graph.edge_count if graph else 0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            errors=errors,
            extra=extra or {},
        )


# ---------------------------------------------------------------------- #
# A: LLM only
# ---------------------------------------------------------------------- #
class DirectLLMSystem(EvalSystem):
    """Answer directly from the query, with no workspace at all."""

    name = "A: llm_only"
    description = "Single synthesizer call; no retrieval, no graph."

    def run_case(self, case: EvalCase) -> EvalRun:
        started = time.perf_counter()
        llm = self._new_llm()
        agent = SynthesizerAgent(llm, self.settings)
        empty = ReasoningContext(task_description=case.query)
        result = agent.run(case.query, empty)
        errors = [] if result.succeeded else [result.error or "synthesizer failed"]
        return self._finalize(
            case,
            answer=result.answer,
            graph=None,
            contexts=[empty],
            total_tokens=result.tokens_used,
            agent_calls=1,
            started=started,
            errors=errors,
        )


# ---------------------------------------------------------------------- #
# B: flat context
# ---------------------------------------------------------------------- #
class FlatContextSystem(EvalSystem):
    """Dump every workspace unit into the prompt, with no typed relations."""

    name = "B: flat_context"
    description = "All units, untyped, no graph expansion; vector store unused."

    def run_case(self, case: EvalCase) -> EvalRun:
        started = time.perf_counter()
        llm = self._new_llm()
        graph = self._new_graph(llm)
        _contents, errors = self._populate(graph, case)
        # Insertion order is deterministic; wall-clock timestamps are not.
        units = graph.nodes()
        context = ReasoningContext(
            task_description=case.query,
            items=self._context_items(units),
            token_estimate=sum(estimate_item_tokens(u.content) for u in units),
        )
        agent = SynthesizerAgent(llm, self.settings)
        result = agent.run(case.query, context)
        if not result.succeeded:
            errors.append(result.error or "synthesizer failed")
        return self._finalize(
            case,
            answer=result.answer,
            graph=graph,
            contexts=[context],
            total_tokens=result.tokens_used,
            agent_calls=1,
            started=started,
            errors=errors,
        )


# ---------------------------------------------------------------------- #
# C: vector RAG
# ---------------------------------------------------------------------- #
class VectorRAGSystem(EvalSystem):
    """Semantic top-k only: no edge walking, stance or provenance."""

    name = "C: vector_rag"
    description = "Cosine top-k seeds only; no relational expansion."

    def run_case(self, case: EvalCase) -> EvalRun:
        started = time.perf_counter()
        llm = self._new_llm()
        graph = self._new_graph(llm)
        _contents, errors = self._populate(graph, case)

        context = ReasoningContext(task_description=case.query)
        if graph.node_count:
            try:
                query_vector = graph.embedder.embed([case.query])[0]
                hits = graph.store.search(
                    query_vector,
                    top_k=self.settings.top_k,
                    min_score=self.settings.cosine_floor,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vector search failed ({type(exc).__name__}): {exc}")
                hits = []
            units = [graph.get_unit(hit.id) for hit in hits if graph.has_node(hit.id)]
            context = ReasoningContext(
                task_description=case.query,
                items=self._context_items(units),
                token_estimate=sum(estimate_item_tokens(u.content) for u in units),
            )

        agent = SynthesizerAgent(llm, self.settings)
        result = agent.run(case.query, context)
        if not result.succeeded:
            errors.append(result.error or "synthesizer failed")
        return self._finalize(
            case,
            answer=result.answer,
            graph=graph,
            contexts=[context],
            total_tokens=result.tokens_used,
            agent_calls=1,
            started=started,
            errors=errors,
        )


# ---------------------------------------------------------------------- #
# D: full DRLW
# ---------------------------------------------------------------------- #
class OrchestratorSystem(EvalSystem):
    """Full pipeline: plan, typed subgraph retrieval, workers, synthesis."""

    name = "D: drlw"
    description = "Full DRLW orchestration with typed relational subgraphs."

    def run_case(self, case: EvalCase) -> EvalRun:
        started = time.perf_counter()
        llm = self._new_llm()
        graph = self._new_graph(llm)
        _contents, errors = self._populate(graph, case)
        orchestrator = build_orchestrator(
            self.settings, llm=llm, graph=graph
        )
        result = orchestrator.run(case.query)
        errors.extend(result.errors)
        return self._finalize(
            case,
            answer=result.final_answer,
            graph=graph,
            contexts=result.contexts,
            total_tokens=result.metrics.total_tokens,
            agent_calls=result.metrics.agent_calls,
            started=started,
            errors=errors,
            extra={
                "rounds": result.metrics.rounds,
                "nodes_added": result.metrics.nodes_added,
                "edges_added": result.metrics.edges_added,
                "answer_grounded": result.extra.get("answer_grounded", False),
            },
        )


class NoExpansionSystem(OrchestratorSystem):
    """Ablation: full orchestration, but retrieval cannot walk edges."""

    name = "ablation: no_expansion"
    description = "DRLW with expansion_depth=0 (semantic seeds only)."

    def run_case(self, case: EvalCase) -> EvalRun:
        started = time.perf_counter()
        llm = self._new_llm()
        graph = self._new_graph(llm)
        _contents, errors = self._populate(graph, case)
        retriever = SubgraphRetriever(
            graph,
            self.settings,
            expansion_depth=0,
            seed_k=self.settings.top_k * 2,
        )
        orchestrator: Orchestrator = build_orchestrator(
            self.settings, llm=llm, graph=graph, retriever=retriever
        )
        result = orchestrator.run(case.query)
        errors.extend(result.errors)
        return self._finalize(
            case,
            answer=result.final_answer,
            graph=graph,
            contexts=result.contexts,
            total_tokens=result.metrics.total_tokens,
            agent_calls=result.metrics.agent_calls,
            started=started,
            errors=errors,
            extra={
                "rounds": result.metrics.rounds,
                "nodes_added": result.metrics.nodes_added,
                "edges_added": result.metrics.edges_added,
                "answer_grounded": result.extra.get("answer_grounded", False),
            },
        )


def default_systems(
    settings: Settings | None = None,
    *,
    include_ablation: bool = True,
) -> list[EvalSystem]:
    """The A-D comparison, plus the no-expansion ablation."""
    systems: list[EvalSystem] = [
        DirectLLMSystem(settings),
        FlatContextSystem(settings),
        VectorRAGSystem(settings),
        OrchestratorSystem(settings),
    ]
    if include_ablation:
        systems.append(NoExpansionSystem(settings))
    return systems
