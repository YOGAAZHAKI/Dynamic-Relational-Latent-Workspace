"""SubgraphRetriever: query -> scored, budgeted ``ReasoningContext``.

What it does (all offline, no LLM calls):

1. **Semantic seeds** — embed the query, ask the graph's vector store
   for the top ``seed_k`` units whose cosine similarity clears
   ``cosine_floor`` (``seed_k`` defaults to ``settings.top_k``).
2. **Graph expansion** — breadth-first from the seeds across typed edges
   (confidence >= ``retrieval_min_edge_confidence``), up to
   ``retrieval_expansion_depth`` hops. A confident CONTRADICTS edge is
   walked just as eagerly as SUPPORTS: conflicts must surface.
3. **Scoring** — for every reached unit::

       score = W_SEMANTIC * cosine(query, unit)
             + W_RELATIONAL * max_edge_signal

   where ``max_edge_signal = edge.confidence * type_weight * hop_decay**(hop-1)``
   over the edges that reached the unit. Seeds keep ``max_edge_signal = 0``
   (they were chosen by similarity; neighbors are ranked by graph signal).
   Relationship-type priorities are configurable
   (``retrieval_type_weights``); CONTRADICTS defaults to the same weight
   as SUPPORTS on purpose.
4. **Stance vs anchor** — the best seed (or an explicit ``anchor_id``)
   is the subgraph's center; each item is labeled supporting /
   contradicting / neutral against it. Contradicting items are never
   dropped silently: budget cuts land them in ``excluded_unit_ids`` with
   ``truncated=True``.
5. **Budget** — items are admitted in score order until the token budget
   (and ``max_units`` cap) is hit. Ties are broken by ascending unit id,
   so results are deterministic run-to-run.

The retriever *reads* the KnowledgeGraph, Embedder and vector store
through their public interfaces — it never mutates them and never
re-implements their jobs (no own index, no own embedding cache).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..config import DEFAULT_TYPE_WEIGHTS, Settings, get_settings
from ..models import (
    ContextItem,
    ContextRelation,
    ProvenanceRef,
    ReasoningContext,
    RelationshipType,
)
from .base import (
    RetrievalError,
    cosine_similarity,
    estimate_item_tokens,
    validate_type_weights,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..graph.workspace import KnowledgeGraph

#: relationship types that mean "this item is consistent with the anchor"
SUPPORTING_TYPES = frozenset(
    {
        RelationshipType.SUPPORTS,
        RelationshipType.REFINES,
        RelationshipType.DEPENDS_ON,
        RelationshipType.REDUNDANT,
    }
)
CONTRADICTING_TYPES = frozenset({RelationshipType.CONTRADICTS})

#: how much provenance a single context item may carry
PROVENANCE_MAX_DEPTH = 2
PROVENANCE_MAX_STEPS = 3


class RetrievalScore(BaseModel):
    """One ranked node with its score breakdown (for tests, demos, UI)."""

    node_id: str
    score: float
    semantic: float  # weighted cosine component
    relational: float  # weighted edge-signal component
    hop: int  # 0 = seed, 1 = neighbor of a seed, ...


class SubgraphRetriever:
    """Retrieve a relevant, scored, budgeted subgraph for a query."""

    def __init__(
        self,
        graph: "KnowledgeGraph",
        settings: Settings | None = None,
        *,
        expansion_depth: int | None = None,
        max_units: int | None = None,
        token_budget: int | None = None,
        seed_k: int | None = None,
        min_edge_confidence: float | None = None,
        weight_semantic: float | None = None,
        weight_relational: float | None = None,
        hop_decay: float | None = None,
        type_weights: dict[str, float] | None = None,
    ) -> None:
        if not hasattr(graph, "embedder") or not hasattr(graph, "store"):
            raise RetrievalError(
                "SubgraphRetriever needs a KnowledgeGraph with .embedder/.store"
            )
        self.graph = graph
        self.settings = settings or getattr(graph, "settings", None) or get_settings()
        s = self.settings

        self.expansion_depth = _as_int(
            expansion_depth, s.retrieval_expansion_depth, "expansion_depth"
        )
        self.max_units = _as_int(max_units, s.retrieval_max_units, "max_units")
        self.token_budget = _as_int(
            token_budget, s.retrieval_token_budget, "token_budget"
        )
        # seed_k defaults to the vector store's top_k; callers that need
        # higher recall (the orchestrator) may widen the seed row.
        self.seed_k = _as_int(seed_k, s.top_k, "seed_k")
        self.min_edge_confidence = float(
            s.retrieval_min_edge_confidence
            if min_edge_confidence is None
            else min_edge_confidence
        )
        self.weight_semantic = float(
            s.retrieval_weight_semantic if weight_semantic is None else weight_semantic
        )
        self.weight_relational = float(
            s.retrieval_weight_relational
            if weight_relational is None
            else weight_relational
        )
        self.hop_decay = float(
            s.retrieval_hop_decay if hop_decay is None else hop_decay
        )
        if not 0.0 <= self.min_edge_confidence <= 1.0:
            raise RetrievalError("min_edge_confidence must be in [0, 1]")

        raw = dict(DEFAULT_TYPE_WEIGHTS)
        raw.update(s.retrieval_type_weights)
        raw.update(type_weights or {})
        self.type_weights = validate_type_weights(raw)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        *,
        anchor_id: str | None = None,
        expansion_depth: int | None = None,
        max_units: int | None = None,
        token_budget: int | None = None,
        seed_k: int | None = None,
    ) -> ReasoningContext:
        """Rank + expand + budget the graph into an agent-ready context."""
        task = _require_query(query)
        depth = _as_int(expansion_depth, self.expansion_depth, "expansion_depth")
        units_cap = _as_int(max_units, self.max_units, "max_units")
        budget = _as_int(token_budget, self.token_budget, "token_budget")
        seed_count = _as_int(seed_k, self.seed_k, "seed_k")

        if self.graph.node_count == 0:
            return ReasoningContext(task_description=task)

        query_vector = self._embed(task)
        seeds = self._semantic_seeds(query_vector, seed_count)
        anchor = self._resolve_anchor(anchor_id, seeds)
        if not seeds:
            return ReasoningContext(task_description=task)

        hop, signal = self._expand(seeds, depth)
        ranked = self._score(query_vector, hop, signal)

        selected, excluded, used, truncated = self._fit_budget(
            ranked, hop, units_cap, budget
        )
        items, supporting, contradicting = self._build_items(selected, anchor)
        return ReasoningContext(
            task_description=task,
            items=items,
            truncated=truncated,
            token_estimate=used,
            anchor_id=anchor,
            supporting_ids=supporting,
            contradicting_ids=contradicting,
            excluded_unit_ids=excluded,
        )

    def rank(self, query: str, *, seed_k: int | None = None) -> list[RetrievalScore]:
        """Public score breakdown: every reached node, best first.

        Deterministic: ties broken by ascending unit id.
        """
        task = _require_query(query)
        if self.graph.node_count == 0:
            return []
        query_vector = self._embed(task)
        seeds = self._semantic_seeds(
            query_vector, _as_int(seed_k, self.seed_k, "seed_k")
        )
        if not seeds:
            return []
        hop, signal = self._expand(seeds, self.expansion_depth)
        return self._score(query_vector, hop, signal)

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    def _embed(self, text: str) -> list[float]:
        try:
            vectors = self.graph.embedder.embed([text])
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        if not vectors or not vectors[0]:
            raise RetrievalError("Embedder returned an empty query vector")
        return vectors[0]

    def _semantic_seeds(
        self, query_vector: list[float], seed_k: int
    ) -> dict[str, float]:
        """node_id -> cosine, best first (deterministic on ties)."""
        try:
            hits = self.graph.store.search(
                query_vector,
                top_k=seed_k,
                min_score=self.settings.cosine_floor,
            )
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc
        seeds: dict[str, float] = {}
        for hit in sorted(hits, key=lambda h: (-h.score, h.id)):
            if self.graph.has_node(hit.id):  # defensive: store must stay in sync
                seeds.setdefault(hit.id, float(hit.score))
        return seeds

    def _resolve_anchor(
        self, anchor_id: str | None, seeds: dict[str, float]
    ) -> str | None:
        if anchor_id is not None:
            if not self.graph.has_node(anchor_id):
                raise RetrievalError(f"anchor_id {anchor_id!r} is not in the graph")
            seeds.setdefault(anchor_id, 0.0)
            return anchor_id
        return next(iter(seeds), None)  # best (-score, id)

    def _expand(
        self, seeds: dict[str, float], depth: int
    ) -> tuple[dict[str, int], dict[str, float]]:
        """BFS from seeds. Returns (hop, best relational signal per node).

        ``hop``: 0 for seeds, 1 for direct neighbors, ... Edges below
        ``min_edge_confidence`` are neither walked nor later displayed.
        Nodes are expanded in ascending id order; signals keep the max,
        so results do not depend on dict ordering.

        Seeds keep a relational signal of 0: they were chosen by query
        similarity, and boosting them again from their own neighbors
        would double-count relevance. Only *expanded* (non-seed) nodes
        earn the weighted edge signal.
        """
        hop: dict[str, int] = {sid: 0 for sid in seeds}
        signal: dict[str, float] = {}
        queue: deque[str] = deque(sorted(seeds))
        while queue:
            current_id = queue.popleft()
            current_hop = hop[current_id]
            if current_hop >= depth:
                continue
            for edge in self._incident_edges(current_id):
                if edge.confidence < self.min_edge_confidence:
                    continue
                other_id = self._other_endpoint(edge, current_id)
                if other_id is None or not self.graph.has_node(other_id):
                    continue
                if other_id in seeds:
                    continue  # seeds stay purely semantic
                # decay per hop away from the seed row (hop 1 -> no decay)
                contribution = (
                    edge.confidence
                    * self.type_weights.get(edge.relationship_type, 1.0)
                    * (self.hop_decay ** current_hop)
                )
                best = signal.get(other_id)
                if best is None or contribution > best:
                    signal[other_id] = contribution
                if other_id not in hop:
                    hop[other_id] = current_hop + 1
                    queue.append(other_id)
        return hop, signal

    def _score(
        self,
        query_vector: list[float],
        hop: dict[str, int],
        signal: dict[str, float],
    ) -> list[RetrievalScore]:
        scores: list[RetrievalScore] = []
        for node_id in sorted(hop):  # deterministic iteration
            unit = self.graph.get_unit(node_id)
            semantic = self.weight_semantic * cosine_similarity(
                query_vector, unit.embedding
            )
            relational = self.weight_relational * signal.get(node_id, 0.0)
            scores.append(
                RetrievalScore(
                    node_id=node_id,
                    score=round(semantic + relational, 6),
                    semantic=round(semantic, 6),
                    relational=round(relational, 6),
                    hop=hop[node_id],
                )
            )
        scores.sort(key=lambda s: (-s.score, s.node_id))
        return scores

    def _fit_budget(
        self,
        ranked: list[RetrievalScore],
        hop: dict[str, int],
        units_cap: int,
        budget: int,
    ) -> tuple[list[RetrievalScore], list[str], int, bool]:
        """Admit items in score order until cap/budget; rest are excluded.

        The first item is always admitted (an over-budget single unit is
        still better than an empty context).
        """
        reached = set(hop)
        selected: list[RetrievalScore] = []
        excluded: list[str] = []
        used = 0
        truncated = False
        for entry in ranked:
            unit = self.graph.get_unit(entry.node_id)
            if len(selected) >= units_cap:
                excluded.append(entry.node_id)
                truncated = True
                continue
            relation_count = sum(
                1
                for e in self._incident_edges(entry.node_id)
                if e.confidence >= self.min_edge_confidence
                and self._other_endpoint(e, entry.node_id) in reached
            )
            cost = estimate_item_tokens(
                unit.content, relation_count, PROVENANCE_MAX_STEPS
            )
            if selected and used + cost > budget:
                excluded.append(entry.node_id)
                truncated = True
                continue
            selected.append(entry)
            used += cost
        return selected, excluded, used, truncated

    def _build_items(
        self,
        selected: list[RetrievalScore],
        anchor: str | None,
    ) -> tuple[list[ContextItem], list[str], list[str]]:
        """Assemble ContextItems with relations, stance and provenance."""
        selected_ids = {entry.node_id for entry in selected}
        items: list[ContextItem] = []
        supporting: list[str] = []
        contradicting: list[str] = []
        for entry in selected:
            unit = self.graph.get_unit(entry.node_id)
            relations: list[ContextRelation] = []
            for edge in self._incident_edges(entry.node_id):
                if edge.confidence < self.min_edge_confidence:
                    continue
                other_id = self._other_endpoint(edge, unit.id)
                if other_id not in selected_ids:
                    continue
                direction = (
                    "incoming" if edge.target_node_id == unit.id else "outgoing"
                )
                relations.append(ContextRelation(edge=edge, direction=direction))
            stance = self._stance(unit.id, anchor)
            items.append(
                ContextItem(
                    unit=unit,
                    relations=relations,
                    stance=stance,
                    provenance=self._provenance_refs(unit.id),
                )
            )
            if unit.id == anchor:
                continue
            if stance == "supporting":
                supporting.append(unit.id)
            elif stance == "contradicting":
                contradicting.append(unit.id)
        return items, supporting, contradicting

    def _stance(self, node_id: str, anchor: str | None) -> str:
        """Supporting / contradicting / neutral *relative to the anchor*.

        Conflicts win over supports when an item does both. Stance uses
        the same incident edges the subgraph displays (confident enough
        to walk), measured in either edge direction.
        """
        if anchor is None or node_id == anchor:
            return "neutral"
        types: set[RelationshipType] = set()
        for edge in self._incident_edges(node_id):
            if edge.confidence < self.min_edge_confidence:
                continue
            if self._other_endpoint(edge, node_id) != anchor:
                continue
            types.add(edge.relationship_type)
        if types & CONTRADICTING_TYPES:
            return "contradicting"
        if types & SUPPORTING_TYPES:
            return "supporting"
        return "neutral"

    def _provenance_refs(self, node_id: str) -> list[ProvenanceRef]:
        """Condensed grounding chain; never raises (provenance is optional)."""
        try:
            steps = self.graph.provenance(node_id, max_depth=PROVENANCE_MAX_DEPTH)
        except Exception:  # noqa: BLE001 - provenance is best-effort metadata
            return []
        refs: list[ProvenanceRef] = []
        for step in steps[:PROVENANCE_MAX_STEPS]:
            refs.append(
                ProvenanceRef(
                    unit_id=step.unit.id,
                    content_preview=step.unit.short(60),
                    relationship_type=step.edge.relationship_type,
                    confidence=step.edge.confidence,
                    depth=step.depth,
                )
            )
        return refs

    # ------------------------------------------------------------------ #
    # graph access (public API only — the graph owns its internals)
    # ------------------------------------------------------------------ #
    def _incident_edges(self, node_id: str):
        seen: dict[str, object] = {}
        for edge in self.graph.incoming_edges(node_id):
            seen[edge.id] = edge
        for edge in self.graph.outgoing_edges(node_id):
            seen[edge.id] = edge
        return sorted(seen.values(), key=lambda e: (-e.confidence, e.id))

    @staticmethod
    def _other_endpoint(edge, node_id: str) -> str | None:
        if edge.source_node_id == node_id:
            return edge.target_node_id
        if edge.target_node_id == node_id:
            return edge.source_node_id
        return None


# ---------------------------------------------------------------------- #
# module helpers
# ---------------------------------------------------------------------- #
def _require_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must be a non-blank string")
    return query.strip()


def _as_int(value: int | None, default: int, name: str) -> int:
    if value is None:
        return int(default)
    value = int(value)
    if value < 0:
        raise RetrievalError(f"{name} must be >= 0, got {value}")
    return value
