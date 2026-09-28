"""`POST /responses`: OpenResponses endpoint.

One path, two content types: JSON when `stream` is false, `text/event-stream` when it is true.
`EventSourceResponse` from `fastapi.sse` only encodes for a path operation that yields, which a
dual-mode endpoint cannot do, so the streaming branch returns a `StreamingResponse` over a
generator that writes the frames itself.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.config.deps import EmbedderDep, SessionDep
from api.routers.openapi import NO_CONTINUATION_EXAMPLE, RESPONSES_DOCS
from api.schemas.responses import CreateResponseBody
from api.services import responses as responses_service

router = APIRouter(tags=["responses"])


# `response_model=None`: the return type is a union of two Response classes, which FastAPI would
# otherwise try to turn into a response model and reject. `RESPONSES_DOCS` documents both bodies
# by hand instead.
@router.post(
    "/responses",
    response_model=None,
    summary="Ask a question about the corpus",
    response_description="A response resource, or an SSE stream of one",
    responses=RESPONSES_DOCS,
)
def create_response(
    body: CreateResponseBody, session: SessionDep, embedder: EmbedderDep
) -> JSONResponse | StreamingResponse:
    """Answer a question about the ingested documents, citing the chunks used.

    ## Example request:

    ```json
    {"model": "small", "input": "What does the corpus say about amyloid beta aggregation?"}
    ```

    ### What happens

    ```
    detect_filters ─► search ─► agent ─► answer
    ```

    `detect_filters` asks the small model for metadata constraints implied by the question
    (author, year, journal) and the search applies them before ranking. The search always runs
    on the question and labels each source `[S1]`, `[S2]`… so the agent can cite them inline; the
    agent may search again with other words.

    ### Citations

    Every `[Sn]` marker in the answer becomes a `url_citation` annotation on the output text,
    pointing at the document and page it came from:

    ```json
    {"type": "url_citation", "url": "/documents/8f3c…#page=12",
     "title": "kumar-etal-amyloid-beta-aggregation.pdf p.12", "start_index": 31, "end_index": 35}
    ```

    Streaming sends each one as a `response.output_text.annotation.added` event as soon as the
    answer is complete enough to place it. That event is the spec's own — a bespoke `references`
    event would be discarded by a compliant client, so there isn't one.

    ### Streaming

    Swagger UI buffers the whole body, so it shows a stream only after it finishes. To watch it
    arrive:

    ```bash
    curl -N localhost:8000/responses -H 'content-type: application/json' \\
      -d '{"model":"small","stream":true,"input":"Summarise the corpus."}'
    ```

    ### Worth knowing

    - **Answers are only as good as what was ingested.** With an empty corpus the agent says it
      found nothing. Upload a PDF via `POST /documents` first.
    - **`model` is a size** — `small`, `medium`, `large` — not a provider model id. The response
      echoes the real id that served it.
    - **No conversation state.** Send the full history in `input`; `previous_response_id` is
      always a 404.
    - **Declaring `tools`** hands control back: the model's call is returned as a `function_call`
      output item and the turn ends there, rather than the server executing it.
    """
    if body.previous_response_id is not None:
        # Nothing is persisted, so the only honest answer is that the id is unknown. Conversation
        # history belongs in `input`.
        return JSONResponse(NO_CONTINUATION_EXAMPLE, status_code=status.HTTP_404_NOT_FOUND)

    if body.stream:
        return StreamingResponse(
            responses_service.stream(body, session, embedder),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(responses_service.create(body, session, embedder).serialize())
