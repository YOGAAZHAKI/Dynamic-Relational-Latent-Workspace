"""Central configuration.

Every knob the system needs lives here and is read from environment
variables (optionally via a `.env` file). The prefix ``DLRW_`` keeps our
variables from colliding with other software.

Usage:
    from latent_workspace.config import get_settings
    settings = get_settings()          # cached, cheap to call anywhere
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    # How much each relationship type boosts a unit's retrieval score.
    # CONTRADICTS scores as high as SUPPORTS on purpose: a retrieval that
    # quietly drops counter-evidence is a broken retrieval.
    "SUPPORTS": 1.0,
    "CONTRADICTS": 1.0,
    "REFINES": 1.0,
    "DEPENDS_ON": 0.9,
    "REDUNDANT": 0.3,
}


class Settings(BaseSettings):
    """All runtime settings. Defaults are chosen for offline development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DLRW_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- run mode -------------------------------------------------------
    mock_mode: bool = True

    # --- LLM -------------------------------------------------------------
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0
    llm_json_retries: int = 2

    # --- embeddings ------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # --- vector retrieval ------------------------------------------------
    top_k: int = 8
    cosine_floor: float = 0.35

    # --- relationship detection -----------------------------------------
    relation_threshold: float = 0.65
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # --- subgraph retrieval (chunk 6) ------------------------------------
    retrieval_expansion_depth: int = 2  # hops away from the semantic seeds
    retrieval_max_units: int = 12  # hard cap on items in a context
    retrieval_token_budget: int = 2500  # soft cap on rendered context size
    retrieval_min_edge_confidence: float = 0.5  # edges below this are not walked
    retrieval_weight_semantic: float = 0.6  # query<->unit cosine weight
    retrieval_weight_relational: float = 0.4  # graph-edge weight
    retrieval_hop_decay: float = 0.5  # score multiplier per extra hop
    retrieval_type_weights: dict[str, float] = DEFAULT_TYPE_WEIGHTS.copy()

    @field_validator("relation_threshold", "cosine_floor")
    @classmethod
    def _score_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return value

    @field_validator(
        "retrieval_min_edge_confidence",
        "retrieval_weight_semantic",
        "retrieval_weight_relational",
        "retrieval_hop_decay",
    )
    @classmethod
    def _retrieval_score_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return value

    @field_validator(
        "retrieval_expansion_depth", "retrieval_max_units", "retrieval_token_budget"
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("retrieval_type_weights")
    @classmethod
    def _type_weight_keys(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if not 0.0 <= weight <= 2.0:
                raise ValueError(
                    f"retrieval_type_weights[{key!r}] must be between 0.0 and 2.0"
                )
        return value

    # --- orchestration ---------------------------------------------------
    max_rounds: int = 4
    max_context_tokens: int = 3000

    # --- storage ---------------------------------------------------------
    data_dir: Path = Path("data")

    @property
    def is_mock(self) -> bool:
        return self.mock_mode

    @property
    def vector_path(self) -> Path:
        return self.data_dir / "vectors.faiss"

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "workspace.sqlite3"

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (cached)."""
    return Settings()
