"""Ingest and inspect documents."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Path, Query, UploadFile, status

from api.config.deps import BlobStoreDep, SessionDep
from api.config.settings import get_api_settings
from api.crud import document as document_crud
from api.routers.openapi import NOT_FOUND_DOCS, UPLOAD_DOCS
from api.schemas import DocumentCreated, DocumentRead, StepStatusRead
from api.services import ingest

router = APIRouter(prefix="/documents", tags=["documents"])

DocumentId = Annotated[UUID, Path(description="From the `document_id` in the upload response.")]


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentCreated,
    summary="Upload a PDF",
    response_description="The new document id, to poll with",
    responses=UPLOAD_DOCS,
)
async def create_document(
    session: SessionDep,
    blobs: BlobStoreDep,
    file: Annotated[UploadFile, File(description="the PDF to ingest")],
) -> DocumentCreated:
    """Accept a PDF and hand it to the pipeline.

    Returns **202** straight away — nothing has been parsed yet. The file lands in blob storage
    and an event goes onto `q.parse`; five workers carry it through parse → chunk → embed →
    citation → index. Poll `GET /documents/{document_id}` until `status` is `completed`, then it
    is searchable.

    ```bash
    curl -F file=@samples/kumar-etal-amyloid-beta-aggregation.pdf localhost:8000/documents
    ```

    Uploading the same bytes twice is cheap and safe: every step is cached on a hash of the
    document content, so a re-import replays cached results instead of re-running the embedding
    model. The new document row is real, though — you get a second id pointing at the same
    content, not a deduplicated one.
    """
    settings = get_api_settings()
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(data)} bytes exceeds limit of {settings.max_upload_bytes}",
        )

    doc = ingest(session, blobs, filename=file.filename or "document.pdf", data=data)
    return DocumentCreated(
        document_id=doc.id, filename=doc.filename, sha256=doc.sha256, status=doc.status
    )


@router.get(
    "",
    response_model=list[DocumentRead],
    summary="List documents",
    response_description="Documents, newest first",
)
def list_documents(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500, description="How many to return.")] = 50,
) -> list[DocumentRead]:
    """Every document, newest first.

    A summary view: `chunk_count` is 0 and `steps` is empty here regardless of the real state.
    Fetch one by id for those — counting chunks and reading the status log per row would make
    listing cost a query per document.
    """
    return [DocumentRead.model_validate(d) for d in document_crud.list_all(session, limit)]


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get one document, with pipeline progress",
    response_description="The document, its per-step history and chunk count",
    responses=NOT_FOUND_DOCS,
)
def get_document(session: SessionDep, document_id: DocumentId) -> DocumentRead:
    """One document, including how far through the pipeline it is.

    This is the endpoint to poll after uploading. Two fields carry the answer:

    - **`status`** — `pending` while queued, `in_progress` while a step is running, `completed`
      once indexed, `failed` if a step raised. It only reaches `completed` after the final step.
    - **`steps`** — one append-only row per step transition, each with `cache_hit` and
      `error_msg`. This is where a `failed` document says what went wrong, and where a re-import
      shows itself: `cache_hit: true` on every step but `index`, which is a sink and never cached.

    `citation` is the metadata extracted by the model — title, authors, journal, DOI. It is
    best-effort: the citation step is a small model and leaves fields null when it cannot find
    them.

    `chunk_count` is how many searchable chunks exist. Zero on a `completed` document means the
    PDF had no extractable text layer.
    """
    doc = document_crud.get(session, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    read = DocumentRead.model_validate(doc)
    read.chunk_count = document_crud.count_chunks(session, document_id)
    read.steps = [
        StepStatusRead.model_validate(s) for s in document_crud.steps(session, document_id)
    ]
    return read


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    response_description="Deleted",
    responses=NOT_FOUND_DOCS,
)
def delete_document(session: SessionDep, blobs: BlobStoreDep, document_id: DocumentId) -> None:
    """Remove a document, its chunks, its status history and its stored blob.

    The step cache is deliberately left behind. It is keyed on document *content*, not id, so it
    stays valid — re-uploading the same PDF after a delete replays the cached steps and costs
    almost nothing.
    """
    doc = document_crud.get(session, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    blob_key = doc.blob_key
    document_crud.remove(session, document_id)
    session.commit()
    blobs.delete(blob_key)
