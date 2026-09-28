"""s5: the sink. Writes document, chunks, and vectors to Postgres.

`exit_queue` is None, so this step acks and stops.
"""

import logging

from core.db import session_scope
from core.hashing import StepConfig
from domain import CitationMetadata, PipelineEvent, StepName, StepStatus
from domain.pg import Chunk, Document
from sqlalchemy import delete

from pipeline.step import PipelineStep, StepResult

log = logging.getLogger(__name__)


class IndexStep(PipelineStep):
    step_name = StepName.INDEX
    step_version = 1
    cacheable = False
    """Writes rows for this document — a side effect, not a value. A cached replay would leave a
    re-uploaded document with no chunks."""

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        chunks = cached[StepName.CHUNK]["chunks"]
        vectors = cached[StepName.EMBED]["embeddings"]
        if not isinstance(chunks, list) or not isinstance(vectors, list):
            raise TypeError("index step needs chunk and embed results as lists")
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        citation_payload = cached.get(StepName.CITATION, {}).get("citation")
        citation = (
            CitationMetadata.model_validate(citation_payload)
            if isinstance(citation_payload, dict)
            else None
        )
        parsed = cached.get(StepName.PARSE, {}).get("page_count")
        page_count = parsed if isinstance(parsed, int) else None

        with session_scope() as session:
            document = session.get(Document, event.document_id)
            if document is None:
                raise ValueError(f"document {event.document_id} missing")

            # Re-import must replace, not append: chunk boundaries shift when the chunker or its
            # config changes, so appending would leave stale rows behind to pollute search.
            session.execute(delete(Chunk).where(Chunk.document_id == event.document_id))

            for chunk, vector in zip(chunks, vectors, strict=True):
                if not isinstance(chunk, dict) or not isinstance(vector, list):
                    raise TypeError("malformed chunk or vector")
                ordinal = chunk.get("ordinal")
                page = chunk.get("page")
                text = chunk.get("text")
                if (
                    not isinstance(ordinal, int)
                    or not isinstance(page, int)
                    or not isinstance(text, str)
                ):
                    raise TypeError(f"bad chunk shape: {chunk!r}")
                embedding: list[float] = []
                for value in vector:
                    if not isinstance(value, (int, float)):
                        raise TypeError(
                            f"embedding contains {type(value).__name__}, expected number"
                        )
                    embedding.append(float(value))
                session.add(
                    Chunk(
                        document_id=event.document_id,
                        ordinal=ordinal,
                        page=page,
                        text=text,
                        embedding=embedding,
                    )
                )

            document.status = StepStatus.COMPLETED
            document.page_count = page_count
            document.citation = citation

        log.info("indexed %s chunks for %s", len(chunks), event.document_id)
        return {"indexed": len(chunks)}

    def step_config(self) -> StepConfig:
        return {"sink": "postgres+pgvector"}
