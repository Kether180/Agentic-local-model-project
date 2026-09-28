"""Ingest: store the bytes, create the row, kick off the pipeline."""

import logging

from core.blobs import BlobStore
from core.hashing import sha256_bytes
from core.rabbit import connect, declare, publish
from domain import PipelineEvent, StepName, StepStatus, queue_name
from domain.pg import Document
from sqlalchemy.orm import Session

from api.crud import document as document_crud

log = logging.getLogger(__name__)


def ingest(session: Session, blobs: BlobStore, filename: str, data: bytes) -> Document:
    """Hash, store, record, publish.

    The hash is computed here because it is the cache key every step keys off — computing it once
    at the front door is what makes a re-import of identical bytes free.
    """
    sha256 = sha256_bytes(data)
    blob_key = f"{sha256}/{filename}"
    blobs.save(blob_key, data)

    doc = document_crud.create(session, filename=filename, sha256=sha256, blob_key=blob_key)
    doc.status = StepStatus.IN_PROGRESS
    session.commit()

    event = PipelineEvent(document_id=doc.id, sha256=sha256, filename=filename, blob_key=blob_key)
    queue = queue_name(StepName.PARSE)
    connection = connect()
    try:
        channel = connection.channel()
        declare(channel, queue)
        publish(channel, queue, event.model_dump_json())
    finally:
        connection.close()

    log.info("queued %s (%s) to %s", doc.id, filename, queue)
    return doc
