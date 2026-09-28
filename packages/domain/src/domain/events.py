"""The queue envelope.

Carries pointers and bookkeeping, never document bytes — the PDF lives in blob storage and the
step outputs live in the cache table. Keeping this small is what lets the whole chain run on one
small message per document.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from domain.enums import StepName, StepStatus


class PipelineEvent(BaseModel):
    document_id: UUID
    sha256: str
    filename: str
    blob_key: str

    step_name: StepName | None = None
    step_history: list[StepName] = Field(default_factory=list)
    step_status: StepStatus = StepStatus.PENDING
    error_msg: str = ""

    artifacts: dict[StepName, str] = Field(default_factory=dict)
    """step -> cache_key of that step's stored output."""

    def artifact_of(self, step: StepName) -> str | None:
        return self.artifacts.get(step)
