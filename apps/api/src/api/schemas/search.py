from uuid import UUID

from pydantic import BaseModel


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page: int
    text: str
    score: float
    """Cosine similarity in [0, 1] — 1 - cosine_distance, so bigger is better."""


class SearchResults(BaseModel):
    query: str
    hits: list[SearchHit]
