"""Testable UI state (no Streamlit imports).

``WorkspaceSession`` owns the live graph, the LLM client and the
orchestrator, and exposes small, plain-Python helpers the Streamlit app
(and tests) can render. Keeping this layer Streamlit-free means the UI
logic is unit-testable without installing the optional frontend.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..graph import KnowledgeGraph, build_workspace
from ..llm import LLMClient, build_llm
from ..models import OrchestratorResult
from ..orchestrator import Orchestrator, build_orchestrator


class UIError(Exception):
    """Recoverable UI/session failure (blank query, missing snapshot...)."""


class WorkspaceSession:
    """Mutable session state behind the Streamlit frontend."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.graph: KnowledgeGraph = (
            orchestrator.graph
            if orchestrator is not None
            else build_workspace(self.settings, llm=self.llm)
        )
        self.orchestrator = orchestrator or build_orchestrator(
            self.settings, llm=self.llm, graph=self.graph
        )
        self.history: list[OrchestratorResult] = []

    # ------------------------------------------------------------------ #
    # runs
    # ------------------------------------------------------------------ #
    def ask(self, query: str, *, max_rounds: int | None = None) -> OrchestratorResult:
        """Run the orchestrator and remember the result."""
        if not isinstance(query, str) or not query.strip():
            raise UIError("Enter a non-blank question first.")
        result = self.orchestrator.run(query, max_rounds=max_rounds)
        self.history.append(result)
        return result

    @property
    def last_result(self) -> OrchestratorResult | None:
        return self.history[-1] if self.history else None

    def reset(self) -> None:
        """Discard the workspace and all run history."""
        self.history.clear()
        self.graph = build_workspace(self.settings, llm=self.llm)
        self.orchestrator = build_orchestrator(
            self.settings, llm=self.llm, graph=self.graph
        )

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path | None = None) -> Path:
        """Save graph snapshot + vectors (default: ``<data_dir>/graph.json``)."""
        target = Path(path) if path is not None else self.settings.graph_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return self.graph.save(target)

    def load(self, path: str | Path | None = None) -> None:
        """Restore a snapshot into this session."""
        source = Path(path) if path is not None else self.settings.graph_path
        if not source.exists():
            raise UIError(f"Snapshot not found: {source}")
        self.graph = KnowledgeGraph.load(
            source,
            embedder=self.graph.embedder,
            detector=self.graph.detector,
            settings=self.settings,
        )
        self.orchestrator = build_orchestrator(
            self.settings, llm=self.llm, graph=self.graph
        )
        self.history.clear()

    # ------------------------------------------------------------------ #
    # view helpers (plain dicts -> st.dataframe)
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, object]:
        return {
            "mode": "mock (offline)" if self.settings.is_mock else "live",
            "model": getattr(self.llm, "model_name", "?"),
            "embedder": getattr(self.graph.embedder, "model_name", "?"),
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
            "runs": len(self.history),
            "graph_path": str(self.settings.graph_path),
        }

    def graph_rows(self) -> list[dict[str, object]]:
        return [
            {
                "id": unit.id,
                "type": unit.information_type.value,
                "agent": unit.agent_id.value,
                "confidence": round(unit.confidence, 2),
                "content": unit.content,
            }
            for unit in sorted(self.graph.nodes(), key=lambda u: u.timestamp)
        ]

    def edge_rows(self) -> list[dict[str, object]]:
        units = {unit.id: unit for unit in self.graph.nodes()}
        rows = []
        for edge in sorted(self.graph.edges(), key=lambda e: e.created_at):
            rows.append(
                {
                    "source": units[edge.source_node_id].content,
                    "relationship": edge.relationship_type.value,
                    "target": units[edge.target_node_id].content,
                    "confidence": round(edge.confidence, 2),
                    "rationale": edge.evidence_or_rationale,
                }
            )
        return rows

    def context_rows(self, result: OrchestratorResult) -> list[dict[str, object]]:
        rows = []
        for index, context in enumerate(result.contexts, start=1):
            for item in context.items:
                rows.append(
                    {
                        "context": index,
                        "unit_id": item.unit.id,
                        "type": item.unit.information_type.value,
                        "stance": item.stance,
                        "confidence": round(item.unit.confidence, 2),
                        "relations": ", ".join(item.relation_labels) or "-",
                        "content": item.unit.content,
                    }
                )
        return rows

    def subtask_rows(self, result: OrchestratorResult) -> list[dict[str, object]]:
        return [
            {
                "id": st.id,
                "agent": st.assigned_agent.value,
                "status": st.status.value,
                "units": len(st.unit_ids),
                "description": st.description,
                "notes": st.notes,
            }
            for st in result.subtasks
        ]
