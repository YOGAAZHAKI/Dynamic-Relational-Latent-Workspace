"""FAISS-backed vector store.

Design decisions (and why)
--------------------------
- ``IndexFlatIP`` = exact inner-product search. Because every vector is
  **L2-normalized before indexing**, inner product == cosine similarity,
  so scores come back as proper cosine values in [-1, 1].
- Stable id mapping: FAISS only knows row numbers (0, 1, 2, ...), so we
  keep ``_ids[row] = "<iu_xxxxxxxx>"`` and a reverse dict. Persisted to a
  JSON sidecar file next to the index, so ids survive restarts.
- Duplicates: re-adding an existing id is skipped (units are embedded
  once, so "first wins" is safe and keeps runs idempotent).
- Removal: ``IndexFlat`` has no cheap delete, so we reconstruct + rebuild.
  Perfectly fine at prototype scale (thousands of vectors).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .base import VectorHit, VectorStoreError

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised only without faiss installed
    faiss = None

_EPS = 1e-12


def ids_sidecar_path(path: str | Path) -> Path:
    """Where the id mapping lives: ``vectors.faiss`` -> ``vectors.faiss.ids.json``."""
    return Path(str(path) + ".ids.json")


class FaissVectorStore:
    """Cosine similarity search over L2-normalized vectors (FAISS)."""

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise VectorStoreError(
                "faiss-cpu is not installed. Install it: python -m pip install faiss-cpu"
            )
        if dim <= 0:
            raise VectorStoreError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self._index = faiss.IndexFlatIP(self.dim)
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}

    # -- introspection ----------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def has(self, node_id: str) -> bool:
        return node_id in self._id_to_row

    # -- internals ---------------------------------------------------------
    def _as_matrix(self, vectors: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise VectorStoreError(f"Vectors are not numeric: {exc}") from exc
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise VectorStoreError(f"Expected a 2-D matrix, got shape {matrix.shape}")
        return np.ascontiguousarray(matrix)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] != self.dim:
            raise VectorStoreError(
                f"Dimension mismatch: store expects {self.dim}, got {matrix.shape[1]}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, _EPS, out=norms)   # zero vector -> stays zero
        return (matrix / norms).astype(np.float32)

    # -- write --------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> int:
        """Add vectors under given ids. Returns count actually added."""
        ids = list(ids)
        if len(ids) != len(vectors):
            raise VectorStoreError(
                f"ids ({len(ids)}) and vectors ({len(vectors)}) length mismatch"
            )
        if len(ids) == 0:
            return 0

        matrix = self._as_matrix(vectors)
        if matrix.shape[1] != self.dim:
            raise VectorStoreError(
                f"Dimension mismatch: store expects {self.dim}, got {matrix.shape[1]}"
            )

        # dedupe: first occurrence in this batch wins, existing ids skipped
        keep_rows: list[int] = []
        keep_ids: list[str] = []
        seen: set[str] = set()
        for row, node_id in enumerate(ids):
            if not node_id or node_id in self._id_to_row or node_id in seen:
                continue
            seen.add(node_id)
            keep_rows.append(row)
            keep_ids.append(node_id)

        if not keep_rows:
            return 0

        normalized = self._normalize(matrix[keep_rows])
        self._index.add(normalized)
        for node_id in keep_ids:
            self._id_to_row[node_id] = len(self._ids)
            self._ids.append(node_id)
        return len(keep_ids)

    def remove(self, node_id: str) -> bool:
        """Remove a vector (rebuilds the flat index). False if unknown."""
        if node_id not in self._id_to_row:
            return False
        total = int(self._index.ntotal)
        keep_positions = [i for i, existing in enumerate(self._ids) if existing != node_id]

        new_index = faiss.IndexFlatIP(self.dim)
        if keep_positions and total > 0:
            all_vectors = self._index.reconstruct_n(0, total)
            kept = np.ascontiguousarray(all_vectors[keep_positions])
            new_index.add(kept)

        self._index = new_index
        self._ids = [self._ids[i] for i in keep_positions]
        self._id_to_row = {nid: i for i, nid in enumerate(self._ids)}
        return True

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        self._ids = []
        self._id_to_row = {}

    # -- search ---------------------------------------------------------------
    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[VectorHit]:
        """Nearest neighbors of ``query`` (cosine scores, best first)."""
        if self.size == 0:
            return []
        if top_k <= 0:
            return []

        query_matrix = self._as_matrix([list(query)])
        query_matrix = self._normalize(query_matrix)

        k = min(int(top_k), self.size)
        scores, positions = self._index.search(query_matrix, k)

        hits: list[VectorHit] = []
        for score, position in zip(scores[0], positions[0]):
            position = int(position)
            if position < 0:  # FAISS pads with -1 when fewer results exist
                continue
            value = float(score)
            if min_score is not None and value < min_score:
                continue
            hits.append(VectorHit(id=self._ids[position], score=value))
        return hits

    # -- persistence ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the index + id sidecar (both must be kept together)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        sidecar = ids_sidecar_path(path)
        sidecar.write_text(
            json.dumps({"dim": self.dim, "ids": self._ids}, indent=0),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        """Load a store previously written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise VectorStoreError(f"Index file not found: {path}")
        sidecar = ids_sidecar_path(path)
        if not sidecar.exists():
            raise VectorStoreError(
                f"Id-mapping sidecar missing: {sidecar} (index and sidecar "
                "must stay together)"
            )
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Corrupt id sidecar {sidecar}: {exc}") from exc

        index = faiss.read_index(str(path))
        stored_ids = payload.get("ids")
        stored_dim = payload.get("dim")
        if not isinstance(stored_ids, list) or not isinstance(stored_dim, int):
            raise VectorStoreError(f"Malformed id sidecar: {sidecar}")
        if stored_dim != int(index.d):
            raise VectorStoreError(
                f"Dim mismatch: index says {index.d}, sidecar says {stored_dim}"
            )
        if len(stored_ids) != int(index.ntotal):
            raise VectorStoreError(
                f"Row mismatch: index has {int(index.ntotal)} vectors but "
                f"sidecar lists {len(stored_ids)} ids"
            )

        store = cls(dim=stored_dim)
        store._index = index
        store._ids = [str(x) for x in stored_ids]
        store._id_to_row = {nid: i for i, nid in enumerate(store._ids)}
        if len(store._id_to_row) != len(store._ids):
            raise VectorStoreError("Duplicate ids found in sidecar")
        return store
