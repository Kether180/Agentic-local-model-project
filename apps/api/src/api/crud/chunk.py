"""Vector search over chunks."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from domain.pg import Chunk, Document
from sqlalchemy import select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    # Import-time only: `crud` must not depend on the agent at runtime, and the agent's state
    # module imports `ChunkHit` from here.
    from api.services.agent.filters import AutoFilter


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    """One entry of a document's references section, cited from inside a chunk."""

    number: int
    text: str
    page: int


@dataclass(frozen=True, slots=True)
class CitedClaim:
    """A sentence of a chunk and the references its citation mark resolves to."""

    text: str
    sources: tuple[ReferenceSource, ...]


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """One search result, narrowed at the boundary.

    pgvector's `cosine_distance` is untyped, so the raw `Row` carries `Any` into every caller.
    Converting here keeps that inside this function.
    """

    chunk_id: UUID
    document_id: UUID
    filename: str
    page: int
    text: str
    distance: float
    claims: tuple[CitedClaim, ...] = ()
    """The chunk's cited sentences, resolved to its references section. Empty without marks."""

    @property
    def score(self) -> float:
        """Cosine similarity in [0, 1] — bigger is better."""
        return 1.0 - self.distance


def search(
    session: Session,
    vector: list[float],
    limit: int,
    filters: "AutoFilter | None" = None,
) -> list[ChunkHit]:
    """Nearest chunks by cosine distance, served by the HNSW index on `chunk.embedding`.

    `filters` narrows to documents whose citation metadata matches before ranking, which is what
    the agent's autofilter step produces. `GET /search` passes none and gets the original query.
    """
    distance = Chunk.embedding.cosine_distance(vector).label("distance")
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.filename, Chunk.page, Chunk.text, distance)
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(limit)
    )
    if filters is not None:
        stmt = stmt.where(*filters.conditions())
    return [
        ChunkHit(
            chunk_id=row[0],
            document_id=row[1],
            filename=row[2],
            page=row[3],
            text=row[4],
            distance=float(row[5]),
        )
        for row in session.execute(stmt).all()
    ]


def page_texts(session: Session, document_ids: set[UUID]) -> dict[UUID, dict[int, str]]:
    """Each document's pages, rebuilt from its chunks in order.

    Chunks overlap, so a line near a chunk edge can appear twice. Callers parse this text, not
    display it, and the reference parser keeps the longest copy of a repeated entry.
    """
    if not document_ids:
        return {}
    stmt = (
        select(Chunk.document_id, Chunk.page, Chunk.text)
        .where(Chunk.document_id.in_(document_ids))
        .order_by(Chunk.document_id, Chunk.ordinal)
    )
    pages: dict[UUID, dict[int, str]] = {}
    for document_id, page, text in session.execute(stmt).tuples():
        by_page = pages.setdefault(document_id, {})
        by_page[page] = f"{by_page[page]}\n{text}" if page in by_page else text
    return pages
