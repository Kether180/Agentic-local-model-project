"""Document queries. Function-based, session as the first argument."""

from uuid import UUID

from domain import StepStatus
from domain.pg import Chunk, Document, PipelineStatus
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session


def create(session: Session, filename: str, sha256: str, blob_key: str) -> Document:
    document = Document(
        filename=filename, sha256=sha256, blob_key=blob_key, status=StepStatus.PENDING
    )
    session.add(document)
    session.flush()
    return document


def get(session: Session, document_id: UUID) -> Document | None:
    return session.get(Document, document_id)


def list_all(session: Session, limit: int = 50) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit)
    return list(session.scalars(stmt).all())


def steps(session: Session, document_id: UUID) -> list[PipelineStatus]:
    stmt = (
        select(PipelineStatus)
        .where(PipelineStatus.document_id == document_id)
        .order_by(PipelineStatus.created_at)
    )
    return list(session.scalars(stmt).all())


def count_chunks(session: Session, document_id: UUID) -> int:
    stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    return session.scalar(stmt) or 0


def remove(session: Session, document_id: UUID) -> None:
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    session.execute(delete(PipelineStatus).where(PipelineStatus.document_id == document_id))
    session.execute(delete(Document).where(Document.id == document_id))
