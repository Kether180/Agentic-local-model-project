"""Raw vector search over chunks."""

from typing import Annotated

from fastapi import APIRouter, Query

from api.config.deps import EmbedderDep, SessionDep
from api.config.settings import get_api_settings
from api.crud import chunk as chunk_crud
from api.schemas import SearchHit, SearchResults

router = APIRouter(prefix="/search", tags=["search"])


@router.get(
    "",
    response_model=SearchResults,
    summary="Search the corpus for chunks",
    response_description="Matching chunks, most similar first",
)
def search(
    session: SessionDep,
    embedder: EmbedderDep,
    q: Annotated[
        str,
        Query(
            min_length=1,
            description="Natural language, not keywords — the query is embedded, not matched.",
            examples=["amyloid beta aggregation"],
        ),
    ],
    limit: Annotated[
        int | None,
        Query(ge=1, description="Defaults to 10, capped at 100."),
    ] = None,
) -> SearchResults:
    """Embed the query with the same model the pipeline used, then kNN over chunks.

    The retrieval layer on its own, with no model writing prose over it. `POST /responses` calls
    exactly this and asks a model to answer from the results — so if an answer looks wrong, run
    the question through here to see whether retrieval or the model is at fault.

    ```bash
    curl "localhost:8000/search?q=amyloid+beta+aggregation&limit=5"
    ```

    `score` is cosine similarity in `[0, 1]`, bigger is better. Treat it as a ranking, not a
    threshold: what counts as a good score depends on the embedding model, and nothing here is
    filtered out by score. A query about something absent from the corpus still returns the
    nearest chunks, just with lower scores.

    Results are chunks, not documents — one document contributes many, so a count of hits is
    never a count of documents. `document_id` and `page` locate each one.

    Only `completed` documents are represented: chunks are written by the final pipeline step.
    """
    settings = get_api_settings()
    top_k = min(limit or settings.search_limit_default, settings.search_limit_max)

    vector = embedder.embed(q)
    hits = chunk_crud.search(session, vector, top_k)
    return SearchResults(
        query=q,
        hits=[
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                page=hit.page,
                text=hit.text,
                score=hit.score,
            )
            for hit in hits
        ],
    )
