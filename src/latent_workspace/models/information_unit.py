"""InformationUnit: the atom of the workspace.

Every statement an agent produces is normalized into this structure.
It carries an optional ``embedding`` (a list of floats) which Chunk 3
fills in via sentence-transformers. Until then ``embedding`` is ``None``
and the unit is still perfectly valid — that is why embeddings are
optional: the unit exists *before* it is vectorized.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .enums import AgentRole, InformationType


def new_id(prefix: str) -> str:
    """Create a short, unique, human-scannable id like ``iu_9f2c1ab4``."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (naive datetimes are forbidden here)."""
    return datetime.now(timezone.utc)


class InformationUnit(BaseModel):
    """One piece of structured information produced by an agent.

    Fields
    ------
    id:
        Unique identifier, assigned automatically (``iu_xxxxxxxx``).
    content:
        The actual statement. Single self-contained sentence/paragraph.
    information_type:
        What kind of claim it is (HYPOTHESIS, EVIDENCE, ...).
    agent_id:
        Which agent produced it (provenance).
    embedding:
        Latent vector as a plain list of floats. ``None`` until
        the embedding stage has run. Length should equal the configured
        embedding dimension (validated by the embedder, not here).
    confidence:
        How confident the producing agent was, in [0.0, 1.0].
    timestamp:
        Creation time, timezone-aware UTC, assigned automatically.
    metadata:
        Free-form extras: sources, subtask id, round number, token usage...
    """

    id: str = Field(default_factory=lambda: new_id("iu"))
    content: str = Field(min_length=1, max_length=4000)
    information_type: InformationType
    agent_id: AgentRole
    embedding: list[float] | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content", mode="after")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def embedding_dim(self) -> int | None:
        return len(self.embedding) if self.embedding is not None else None

    def set_embedding(self, vector: list[float]) -> None:
        """Attach a latent vector to this unit (validated as floats)."""
        if not vector:
            raise ValueError("Embedding vector must not be empty")
        self.embedding = [float(x) for x in vector]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe dict for persistence (timestamps become ISO strings)."""
        data = self.model_dump(mode="json")
        return data

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> "InformationUnit":
        """Rebuild a unit from :meth:`to_record` output."""
        return cls.model_validate(data)

    def short(self, width: int = 80) -> str:
        """One-line preview for logs and the Streamlit UI."""
        text = self.content if len(self.content) <= width else self.content[: width - 1] + "..."
        return f"[{self.id}] ({self.information_type.value}) {text}"
