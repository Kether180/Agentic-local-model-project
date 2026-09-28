"""Request/response contracts. Never returns an ORM row directly."""

from datetime import datetime
from uuid import UUID

from domain import CitationMetadata, StepName, StepStatus
from pydantic import BaseModel, ConfigDict


class DocumentCreated(BaseModel):
    document_id: UUID
    filename: str
    sha256: str
    status: StepStatus


class StepStatusRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step_name: StepName
    step_status: StepStatus
    cache_hit: bool
    error_msg: str
    created_at: datetime


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    sha256: str
    status: StepStatus
    page_count: int | None
    citation: CitationMetadata | None
    created_at: datetime
    chunk_count: int = 0
    steps: list[StepStatusRead] = []
