"""Postgres schema.

Four tables: the document and its chunks (the searchable output), the step cache (what makes a
re-import cheap), and an append-only status log.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from pydantic import JsonValue
from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from domain.citation import CitationMetadata
from domain.enums import StepName, StepStatus
from domain.pg.base import Base, created_at_column, uuid_pk
from domain.pg.types import PydanticJSONB

EMBED_DIM = 1024
"""The one definition of the vector width — `core` has no copy, it derives the number per model
from `Embedder.dimensions`.

This is schema, not config: changing embedding model to one with a different width needs a
migration, not an env var. `tests/test_workspace.py` asserts the configured embedder agrees with
this, which is the seam where the two independent packages are reconciled."""


class Document(Base):
    __tablename__ = "document"

    id: Mapped[uuid.UUID] = uuid_pk()
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    blob_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[StepStatus] = mapped_column(String(32), default=StepStatus.PENDING)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    citation: Mapped[CitationMetadata | None] = mapped_column(
        PydanticJSONB(CitationMetadata), default=None
    )
    created_at: Mapped[datetime] = created_at_column()

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_document_ordinal"),
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = created_at_column()

    document: Mapped[Document] = relationship(back_populates="chunks")


class StepCache(Base):
    """One row per (document content, step, step version, step config).

    Keyed on the document's content hash rather than its id, so changed bytes can never serve an
    artifact built from the old ones — see `core.hashing.cache_key` for what else goes into the
    key and why.
    """

    __tablename__ = "step_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_sha: Mapped[str] = mapped_column(String(64), index=True)
    step_name: Mapped[StepName] = mapped_column(String(32))
    step_version: Mapped[int] = mapped_column(Integer)
    config_hash: Mapped[str] = mapped_column(String(32))
    result: Mapped[dict[str, JsonValue]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_at_column()


class PipelineStatus(Base):
    """Append-only audit row, written on every step transition."""

    __tablename__ = "pipeline_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    step_name: Mapped[StepName] = mapped_column(String(32))
    step_status: Mapped[StepStatus] = mapped_column(String(32))
    cache_hit: Mapped[bool] = mapped_column(default=False)
    error_msg: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = created_at_column()
