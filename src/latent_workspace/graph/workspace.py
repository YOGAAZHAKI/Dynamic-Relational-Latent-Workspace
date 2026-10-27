"""KnowledgeGraph: the Dynamic Relational Latent Workspace.

Integrates (by dependency injection) the three collaborators built so far:

    Embedder         (Chunk 3)  text  -> vector
    FaissVectorStore (Chunk 3)  vector -> candidate ids (cosine >= floor)
    RelationDetector (Chunk 4)  pair  -> {type, confidence, rationale}

and maintains the graph itself: nodes = InformationUnits, edges =
InformationEdges (typed, confidence-scored, direction
``source --RELATIONSHIP--> target``, i.e. the newer unit points at the
older unit it acts on).

IMPORTANT: the graph stores *predicted* relationships with confidence.
It does NOT prove any statement is factually correct. Retrieval,
prompts and answers must keep surfacing confidences and contradictions
rather than treating edges as truth.

Failure policy (no corrupted state):

- embedding / store-insert failures  -> raise KnowledgeGraphError,
  graph unchanged
- detector failures on one candidate -> recorded in
  ``AddUnitResult.errors``, other candidates still processed, node added
- low-confidence / UNRELATED         -> recorded in
  ``AddUnitResult.skipped``, no edge created
- persistence failures               -> raise KnowledgeGraphError;
  ``load`` validates everything before returning an object
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings, get_settings
from ..embeddings.base import Embedder
from ..models import InformationEdge, InformationUnit, RelationshipType, utc_now
from ..relations.base import RelationDetector, RelationDetectorError, RelationPrediction
from ..vector.base import VectorStoreError
from ..vector.faiss_store import FaissVectorStore

SNAPSHOT_FORMAT = "drlw-graph"
SNAPSHOT_VERSION = 1

#: edge types walked when tracing provenance (what *grounds* a claim).
#: CONTRADICTS is deliberately excluded: counter-evidence is surfaced by
#: ``find_contradicting``, not treated as provenance.
PROVENANCE_TYPES = (
    RelationshipType.SUPPORTS,
    RelationshipType.DEPENDS_ON,
    RelationshipType.REFINES,
)


class KnowledgeGraphError(Exception):
    """Graph construction/validation/persistence failure."""


class AddUnitResult(BaseModel):
    """Report of one incremental ``add_unit()`` update."""

    unit_id: str
    unit: InformationUnit | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    edges_added: list[InformationEdge] = Field(default_factory=list)
    edges_updated: list[InformationEdge] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def added_edge_count(self) -> int:
        return len(self.edges_added)


class ProvenanceStep(BaseModel):
    """One hop in a provenance chain: a unit reached *via* an edge."""

    unit: InformationUnit
    edge: InformationEdge  # the edge pointing INTO the reached unit
    depth: int


class KnowledgeGraph:
    """Incremental relational workspace over an injected embedder,
    vector store and relation detector."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store,
        detector: RelationDetector,
        settings: Settings | None = None,
    ) -> None:
        embed_dim = int(getattr(embedder, "dim", 0) or 0)
        store_dim = int(getattr(vector_store, "dim", 0) or 0)
        if embed_dim <= 0 or store_dim <= 0:
            raise KnowledgeGraphError(
                f"Embedder/store report invalid dims ({embed_dim}/{store_dim})"
            )
        if embed_dim != store_dim:
            raise KnowledgeGraphError(
                f"Embedder dim ({embed_dim}) != vector store dim ({store_dim}); "
                "align DLRW_EMBEDDING_DIM with your embedding model"
            )
        self.embedder = embedder
        self.store = vector_store
        self.detector = detector
        self.settings = settings or get_settings()

        self._nodes: dict[str, InformationUnit] = {}
        self._edges: dict[str, InformationEdge] = {}
        self._incoming: dict[str, list[str]] = {}  # node -> edge ids pointing at it
        self._outgoing: dict[str, list[str]] = {}  # node -> edge ids leaving it
        self._edge_keys: dict[tuple[str, str, RelationshipType], str] = {}

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def nodes(self) -> list[InformationUnit]:
        return list(self._nodes.values())

    def edges(self) -> list[InformationEdge]:
        return list(self._edges.values())

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_unit(self, node_id: str) -> InformationUnit:
        if node_id not in self._nodes:
            raise KnowledgeGraphError(f"Unknown node id: {node_id}")
        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> InformationEdge:
        if edge_id not in self._edges:
            raise KnowledgeGraphError(f"Unknown edge id: {edge_id}")
        return self._edges[edge_id]

    def outgoing_edges(
        self, node_id: str, relationship: RelationshipType | None = None
    ) -> list[InformationEdge]:
        return self._edges_of(self._outgoing.get(node_id, []), relationship)

    def incoming_edges(
        self, node_id: str, relationship: RelationshipType | None = None
    ) -> list[InformationEdge]:
        return self._edges_of(self._incoming.get(node_id, []), relationship)

    def _edges_of(
        self, edge_ids: Iterable[str], relationship: RelationshipType | None
    ) -> list[InformationEdge]:
        result = [self._edges[eid] for eid in edge_ids if eid in self._edges]
        if relationship is not None:
            result = [e for e in result if e.relationship_type is relationship]
        return sorted(result, key=lambda e: e.confidence, reverse=True)

    # ------------------------------------------------------------------ #
    # incremental updates
    # ------------------------------------------------------------------ #
    def add_unit(
        self,
        unit: InformationUnit,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        threshold: float | None = None,
    ) -> AddUnitResult:
        """Add one information unit: embed -> candidates -> classify ->
        threshold -> commit node + valid edges.

        Self-comparison is impossible by construction: candidates are
        searched *before* the node enters the store, with a defensive
        ``id != unit.id`` filter on top.
        """
        if not isinstance(unit, InformationUnit):
            raise KnowledgeGraphError(
                f"add_unit expects an InformationUnit, got {type(unit).__name__}"
            )
        if not unit.content or not unit.content.strip():
            raise KnowledgeGraphError("Refusing to add a unit with blank content")
        if unit.id in self._nodes:
            raise KnowledgeGraphError(
                f"Duplicate node id {unit.id} (nodes are never overwritten)"
            )
        if self.store.has(unit.id):
            raise KnowledgeGraphError(
                f"Vector store already contains {unit.id} but the graph does not "
                "(store/graph desync)"
            )

        top_k = self.settings.top_k if top_k is None else top_k
        min_score = (
            self.settings.cosine_floor if min_score is None else min_score
        )
        threshold = (
            self.settings.relation_threshold if threshold is None else threshold
        )

        # -- 1. embed (nothing mutated yet) -------------------------------
        if unit.embedding is None:
            try:
                vectors = self.embedder.embed([unit.content])
            except Exception as exc:
                raise KnowledgeGraphError(
                    f"Embedding failed for {unit.id} (graph unchanged): {exc}"
                ) from exc
            if not vectors or not vectors[0]:
                raise KnowledgeGraphError(
                    f"Embedder returned no vector for {unit.id} (graph unchanged)"
                )
            unit.set_embedding(vectors[0])
        if len(unit.embedding) != self.store.dim:
            raise KnowledgeGraphError(
                f"Embedding dim {len(unit.embedding)} != store dim "
                f"{self.store.dim} (graph unchanged)"
            )

        # -- 2. candidate retrieval (still read-only) ------------------------
        try:
            hits = self.store.search(
                unit.embedding, top_k=top_k, min_score=min_score
            )
        except VectorStoreError as exc:
            raise KnowledgeGraphError(
                f"Candidate search failed for {unit.id} (graph unchanged): {exc}"
            ) from exc
        candidate_ids = [h.id for h in hits if h.id != unit.id]

        # -- 3. classify every candidate (failures are recorded, not fatal) ---
        result = AddUnitResult(unit_id=unit.id, unit=unit, candidate_ids=candidate_ids)
        accepted_edges: list[InformationEdge] = []

        for candidate_id in candidate_ids:
            candidate = self._nodes.get(candidate_id)
            if candidate is None:
                result.errors.append(
                    f"{candidate_id}: in vector store but not in graph (desync)"
                )
                continue
            try:
                prediction = self.detector.detect(unit.content, candidate.content)
                if not isinstance(prediction, RelationPrediction):
                    raise RelationDetectorError(
                        f"detector returned {type(prediction).__name__}, "
                        "expected RelationPrediction"
                    )
                edge = prediction.to_edge(unit.id, candidate_id, threshold=threshold)
            except RelationDetectorError as exc:
                result.errors.append(f"{candidate_id}: {exc}")
                continue
            except ValidationError as exc:
                result.errors.append(f"{candidate_id}: invalid prediction: {exc}")
                continue
            except Exception as exc:  # one bad pair must not kill the update
                result.errors.append(
                    f"{candidate_id}: detector crash {type(exc).__name__}: {exc}"
                )
                continue

            if edge is None:
                if prediction.is_storable:
                    result.skipped.append(
                        f"{candidate_id}: {prediction.relationship_type.value} "
                        f"conf={prediction.confidence:.2f} below threshold "
                        f"{threshold:.2f}"
                    )
                else:
                    result.skipped.append(
                        f"{candidate_id}: UNRELATED "
                        f"(conf={prediction.confidence:.2f}) -> no edge"
                    )
                continue
            accepted_edges.append(edge)

        # -- 4. commit (order chosen so no failure leaves dangling state) ----
        try:
            self.store.add([unit.id], [unit.embedding])
        except VectorStoreError as exc:
            raise KnowledgeGraphError(
                f"Vector store insert failed for {unit.id} (graph unchanged): {exc}"
            ) from exc

        self._nodes[unit.id] = unit
        self._incoming.setdefault(unit.id, [])
        self._outgoing.setdefault(unit.id, [])
        for edge in accepted_edges:
            self._register_edge(edge)
            result.edges_added.append(edge)

        return result

    def add_units(self, units: Iterable[InformationUnit]) -> list[AddUnitResult]:
        """Batch convenience: sequential incremental adds."""
        return [self.add_unit(unit) for unit in units]

    # ------------------------------------------------------------------ #
    # edge management (public: manual injection, upsert, removal)
    # ------------------------------------------------------------------ #
    def add_edge(self, edge: InformationEdge) -> InformationEdge:
        """Insert (or upgrade) one edge. Endpoints must already exist.

        Upsert rule: same (source, target, type) with a *higher*
        confidence replaces the stored edge (id preserved); otherwise the
        existing edge is kept unchanged.
        """
        if not isinstance(edge, InformationEdge):
            raise KnowledgeGraphError(
                f"add_edge expects an InformationEdge, got {type(edge).__name__}"
            )
        for endpoint in (edge.source_node_id, edge.target_node_id):
            if endpoint not in self._nodes:
                raise KnowledgeGraphError(
                    f"Edge endpoint {endpoint} is not a node in this graph"
                )
        key = (edge.source_node_id, edge.target_node_id, edge.relationship_type)
        existing_id = self._edge_keys.get(key)
        if existing_id is None:
            self._register_edge(edge)
            return edge
        existing = self._edges[existing_id]
        if edge.confidence > existing.confidence:
            upgraded = edge.model_copy(update={"id": existing_id})
            self._edges[existing_id] = upgraded
            return upgraded
        return existing

    def remove_edge(self, edge_id: str) -> bool:
        """Delete one edge (e.g. invalidated later). False if unknown."""
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return False
        if edge_id in self._outgoing.get(edge.source_node_id, []):
            self._outgoing[edge.source_node_id].remove(edge_id)
        if edge_id in self._incoming.get(edge.target_node_id, []):
            self._incoming[edge.target_node_id].remove(edge_id)
        self._edge_keys.pop(
            (edge.source_node_id, edge.target_node_id, edge.relationship_type), None
        )
        return True

    def remove_unit(self, node_id: str) -> bool:
        """Delete a node and every edge touching it, plus its vector."""
        if node_id not in self._nodes:
            return False
        try:
            self.store.remove(node_id)  # vector first: failure leaves graph intact
        except VectorStoreError as exc:
            raise KnowledgeGraphError(
                f"Vector removal failed for {node_id} (graph unchanged): {exc}"
            ) from exc
        for edge_id in list(self._outgoing.get(node_id, [])) + list(
            self._incoming.get(node_id, [])
        ):
            self.remove_edge(edge_id)
        self._nodes.pop(node_id, None)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        return True

    def _register_edge(self, edge: InformationEdge) -> None:
        self._edges[edge.id] = edge
        self._outgoing.setdefault(edge.source_node_id, []).append(edge.id)
        self._incoming.setdefault(edge.target_node_id, []).append(edge.id)
        self._edge_keys[
            (edge.source_node_id, edge.target_node_id, edge.relationship_type)
        ] = edge.id

    # ------------------------------------------------------------------ #
    # queries: supporting / contradicting evidence, provenance
    # ------------------------------------------------------------------ #
    def find_supporting(
        self, node_id: str, *, min_confidence: float | None = None
    ) -> list[tuple[InformationUnit, InformationEdge]]:
        """Units that SUPPORT ``node_id`` (incoming SUPPORTS edges),
        most confident first. Returns (unit, edge) pairs."""
        return self._collect_evidence(
            node_id, RelationshipType.SUPPORTS, "incoming", min_confidence
        )

    def find_contradicting(
        self, node_id: str, *, min_confidence: float | None = None
    ) -> list[tuple[InformationUnit, InformationEdge]]:
        """Units that CONTRADICT ``node_id`` (incoming CONTRADICTS edges),
        most confident first. Returns (unit, edge) pairs."""
        return self._collect_evidence(
            node_id, RelationshipType.CONTRADICTS, "incoming", min_confidence
        )

    def _collect_evidence(
        self,
        node_id: str,
        relationship: RelationshipType,
        direction: str,
        min_confidence: float | None,
    ) -> list[tuple[InformationUnit, InformationEdge]]:
        self.get_unit(node_id)  # raises for unknown ids
        edges = (
            self.incoming_edges(node_id, relationship)
            if direction == "incoming"
            else self.outgoing_edges(node_id, relationship)
        )
        pairs: list[tuple[InformationUnit, InformationEdge]] = []
        for edge in edges:
            if min_confidence is not None and edge.confidence < min_confidence:
                continue
            other_id = (
                edge.source_node_id
                if direction == "incoming"
                else edge.target_node_id
            )
            other = self._nodes.get(other_id)
            if other is not None:
                pairs.append((other, edge))
        return pairs

    def provenance(
        self,
        node_id: str,
        *,
        max_depth: int = 8,
        edge_types: tuple[RelationshipType, ...] = PROVENANCE_TYPES,
    ) -> list[ProvenanceStep]:
        """Walk *backwards* along SUPPORTS / DEPENDS_ON / REFINES edges to
        show what grounds ``node_id``. Breadth-first, cycle-safe (visited
        set), ordered by depth. Does NOT follow CONTRADICTS: use
        :meth:`find_contradicting` for counter-evidence.

        Note: steps are *predicted* links — provenance shows how the
        workspace believes a claim was built, not verified truth.
        """
        self.get_unit(node_id)
        steps: list[ProvenanceStep] = []
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.incoming_edges(current_id):
                if edge.relationship_type not in edge_types:
                    continue
                source_id = edge.source_node_id
                if source_id in visited:
                    continue
                source = self._nodes.get(source_id)
                if source is None:
                    continue
                visited.add(source_id)
                steps.append(
                    ProvenanceStep(unit=source, edge=edge, depth=depth + 1)
                )
                queue.append((source_id, depth + 1))
        return steps

    # ------------------------------------------------------------------ #
    # integrity + persistence
    # ------------------------------------------------------------------ #
    def validate_integrity(self) -> None:
        """Raise KnowledgeGraphError if invariants are broken."""
        for edge in self._edges.values():
            if edge.source_node_id not in self._nodes:
                raise KnowledgeGraphError(
                    f"Edge {edge.id} references unknown source "
                    f"{edge.source_node_id}"
                )
            if edge.target_node_id not in self._nodes:
                raise KnowledgeGraphError(
                    f"Edge {edge.id} references unknown target "
                    f"{edge.target_node_id}"
                )
            if edge.source_node_id == edge.target_node_id:
                raise KnowledgeGraphError(f"Self-edge detected: {edge.id}")
        self._validate_vector_sync()

    def _validate_vector_sync(self) -> None:
        vector_ids = set(self.store.ids)
        node_vector_ids = {
            node_id
            for node_id, unit in self._nodes.items()
            if unit.embedding is not None
        }
        missing = vector_ids - node_vector_ids
        unindexed = node_vector_ids - vector_ids
        if missing or unindexed:
            raise KnowledgeGraphError(
                "Vector index out of sync with graph nodes: "
                f"in-index-but-not-embedded={sorted(missing)}, "
                f"embedded-but-not-in-index={sorted(unindexed)}"
            )
        for node_id in node_vector_ids:
            unit = self._nodes[node_id]
            if len(unit.embedding) != self.store.dim:  # type: ignore[arg-type]
                raise KnowledgeGraphError(
                    f"Node {node_id} embedding dim {len(unit.embedding)} != "
                    f"store dim {self.store.dim}"
                )

    @staticmethod
    def _snapshot_vector_path(snapshot_path: Path) -> Path:
        """``graph.json`` -> ``graph.faiss`` (keeps files together)."""
        return snapshot_path.with_suffix(".faiss")

    def save(self, path: str | Path) -> Path:
        """Write a snapshot: graph JSON (atomic) + vector index.

        Returns the snapshot path. Any failure raises
        :class:`KnowledgeGraphError`.
        """
        snapshot_path = Path(path)
        vector_path = self._snapshot_vector_path(snapshot_path)
        try:
            self.validate_integrity()
        except KnowledgeGraphError:
            raise

        payload = {
            "format": SNAPSHOT_FORMAT,
            "version": SNAPSHOT_VERSION,
            "embedding_dim": self.store.dim,
            "saved_at": utc_now().isoformat(),
            "nodes": [unit.to_record() for unit in self._nodes.values()],
            "edges": [edge.to_record() for edge in self._edges.values()],
        }
        try:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise KnowledgeGraphError(f"Snapshot not JSON-serialisable: {exc}") from exc

        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.store.save(vector_path)  # vectors first, snapshot second
            tmp_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, snapshot_path)  # atomic on POSIX and Windows
        except Exception as exc:
            raise KnowledgeGraphError(
                f"Failed to save snapshot to {snapshot_path}: {exc}"
            ) from exc
        return snapshot_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder: Embedder,
        detector: RelationDetector,
        settings: Settings | None = None,
    ) -> "KnowledgeGraph":
        """Restore a snapshot. Validates EVERYTHING before returning, so a
        failed load never yields a half-built graph."""
        snapshot_path = Path(path)
        vector_path = cls._snapshot_vector_path(snapshot_path)
        if not snapshot_path.exists():
            raise KnowledgeGraphError(f"Snapshot not found: {snapshot_path}")

        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeGraphError(
                f"Corrupt or unreadable snapshot {snapshot_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise KnowledgeGraphError("Snapshot root must be a JSON object")
        if payload.get("format") != SNAPSHOT_FORMAT:
            raise KnowledgeGraphError(
                f"Unknown snapshot format: {payload.get('format')!r}"
            )
        if payload.get("version") != SNAPSHOT_VERSION:
            raise KnowledgeGraphError(
                f"Unsupported snapshot version: {payload.get('version')!r} "
                f"(expected {SNAPSHOT_VERSION})"
            )
        dim = payload.get("embedding_dim")
        if not isinstance(dim, int) or dim <= 0:
            raise KnowledgeGraphError(f"Invalid embedding_dim in snapshot: {dim!r}")
        if int(getattr(embedder, "dim", 0)) != dim:
            raise KnowledgeGraphError(
                f"Snapshot dim {dim} does not match embedder dim "
                f"{getattr(embedder, 'dim', None)}; use the same "
                "DLRW_EMBEDDING_MODEL/DIM to load"
            )

        node_records = payload.get("nodes")
        edge_records = payload.get("edges")
        if not isinstance(node_records, list) or not isinstance(edge_records, list):
            raise KnowledgeGraphError("Snapshot 'nodes' and 'edges' must be lists")

        # -- parse nodes ---------------------------------------------------
        nodes: dict[str, InformationUnit] = {}
        for record in node_records:
            try:
                unit = InformationUnit.from_record(record)
            except ValidationError as exc:
                raise KnowledgeGraphError(f"Invalid node record: {exc}") from exc
            if unit.id in nodes:
                raise KnowledgeGraphError(f"Duplicate node id in snapshot: {unit.id}")
            if unit.embedding is not None and len(unit.embedding) != dim:
                raise KnowledgeGraphError(
                    f"Node {unit.id} embedding dim {len(unit.embedding)} != "
                    f"snapshot dim {dim}"
                )
            nodes[unit.id] = unit

        # -- parse edges (validators reject self-edges / UNRELATED) ---------
        edges: dict[str, InformationEdge] = {}
        edge_keys: dict[tuple[str, str, RelationshipType], str] = {}
        for record in edge_records:
            try:
                edge = InformationEdge.from_record(record)
            except ValidationError as exc:
                raise KnowledgeGraphError(f"Invalid edge record: {exc}") from exc
            if edge.id in edges:
                raise KnowledgeGraphError(f"Duplicate edge id in snapshot: {edge.id}")
            for endpoint in (edge.source_node_id, edge.target_node_id):
                if endpoint not in nodes:
                    raise KnowledgeGraphError(
                        f"Edge {edge.id} references unknown node {endpoint}"
                    )
            key = (edge.source_node_id, edge.target_node_id, edge.relationship_type)
            if key in edge_keys:
                raise KnowledgeGraphError(
                    f"Duplicate relationship {key} in snapshot "
                    f"(edge ids {edge_keys[key]!r} and {edge.id!r})"
                )
            edges[edge.id] = edge
            edge_keys[key] = edge.id

        # -- restore vector index + verify sync ------------------------------
        try:
            store = FaissVectorStore.load(vector_path)
        except VectorStoreError as exc:
            raise KnowledgeGraphError(
                f"Failed to restore vector index {vector_path}: {exc}"
            ) from exc
        if store.dim != dim:
            raise KnowledgeGraphError(
                f"Vector index dim {store.dim} != snapshot dim {dim}"
            )
        vector_ids = set(store.ids)
        node_vector_ids = {
            nid for nid, unit in nodes.items() if unit.embedding is not None
        }
        if vector_ids != node_vector_ids:
            raise KnowledgeGraphError(
                "Snapshot and vector index are out of sync: "
                f"in-index-not-in-nodes={sorted(vector_ids - node_vector_ids)}, "
                f"in-nodes-not-in-index={sorted(node_vector_ids - vector_ids)}"
            )

        # -- build the live object only after all validation passed ----------
        graph = cls(embedder, store, detector, settings)
        graph._nodes = nodes
        graph._edges = edges
        graph._edge_keys = edge_keys
        graph._incoming = {nid: [] for nid in nodes}
        graph._outgoing = {nid: [] for nid in nodes}
        for edge in edges.values():
            graph._outgoing[edge.source_node_id].append(edge.id)
            graph._incoming[edge.target_node_id].append(edge.id)
        graph.validate_integrity()
        return graph


def build_workspace(settings: Settings | None = None, *, llm=None) -> KnowledgeGraph:
    """Factory: a fresh workspace wired with the settings-appropriate
    embedder, FAISS store and relation detector (mock in mock mode)."""
    from ..embeddings import build_embedder
    from ..relations import build_relation_detector
    from ..vector import build_vector_store

    s = settings or get_settings()
    embedder = build_embedder(s)
    store = build_vector_store(embedder, s)
    detector = build_relation_detector(llm=llm, settings=s)
    return KnowledgeGraph(embedder, store, detector, s)
