"""Documentation that `/docs` renders: tag blurbs and the response shapes FastAPI cannot infer."""

from typing import TypeAlias

from api.schemas.responses import ResponseResource

ResponseDocs: TypeAlias = dict[int | str, dict[str, object]]

TAGS: list[dict[str, object]] = [
    {
        "name": "documents",
        "description": (
            "Ingest PDFs and watch them move through the pipeline.\n\n"
            "`POST /documents` returns **202** immediately - parsing, chunking, embedding, "
            "citation extraction and indexing happen on five RabbitMQ queues afterwards. Poll "
            "`GET /documents/{id}` until `status` is `completed`; a document is only searchable "
            "once it is."
        ),
    },
    {
        "name": "responses",
        "description": (
            "Ask questions about the corpus and get an answer with citations.\n\n"
            "An [OpenResponses](https://www.openresponses.org/) endpoint backed by a LangGraph "
            "agent: it infers metadata filters from the question, searches the chunk index, and "
            "cites the chunks it used."
        ),
    },
    {
        "name": "search",
        "description": (
            "Raw vector search. Returns ranked chunks, not an answer: use `POST /responses` if "
            "you want prose. Useful for seeing exactly what the agent has to work with."
        ),
    },
    {"name": "health", "description": "Liveness probe."},
]

DESCRIPTION = """\
A document-processing pipeline with a chat endpoint over its output.

### Try it in order

1. **`POST /documents`**: upload a PDF. Returns `202` and a `document_id`; the work happens on
   queues.
2. **`GET /documents/{id}`**: poll until `status` is `completed`. The `steps` array shows each
   pipeline stage and whether it was a cache hit.
3. **`POST /responses`**: ask a question. Set `"stream": true` to watch the answer and its
   citations arrive.
4. **`GET /search`**: the retrieval layer on its own, if you want to see the chunks behind an
   answer.

Re-upload the same PDF and step 2 reports `cache_hit: true` for every stage but the last. The
step cache is keyed on document content, so identical bytes never re-run the embedding model.
"""

SSE_EXAMPLE = """\
event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_8f3c","status":"in_progress",...}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{...}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_a1","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"msg_a1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_a1","output_index":0,"content_index":0,"delta":"Aggregation is "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":5,"item_id":"msg_a1","output_index":0,"content_index":0,"delta":"nucleation-dependent [S2]."}

event: response.output_text.annotation.added
data: {"type":"response.output_text.annotation.added","sequence_number":6,"item_id":"msg_a1","output_index":0,"content_index":0,"annotation_index":0,"annotation":{"type":"url_citation","url":"/documents/8f3c.../#page=12","title":"kumar-etal-amyloid-beta-aggregation.pdf p.12","start_index":31,"end_index":35}}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":7,"item_id":"msg_a1","output_index":0,"content_index":0,"text":"Aggregation is nucleation-dependent [S2]."}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":8,...}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":9,...}

event: response.completed
data: {"type":"response.completed","sequence_number":10,"response":{"id":"resp_8f3c","status":"completed","output":[...],...}}

data: [DONE]
"""

NO_CONTINUATION_EXAMPLE: dict[str, object] = {
    "error": {
        "type": "invalid_request",
        "code": "previous_response_not_found",
        "param": "previous_response_id",
        "message": (
            "This server does not store responses, so a previous response can never be "
            "continued. Send the full conversation in `input` instead."
        ),
    }
}

RESPONSES_DOCS: ResponseDocs = {
    200: {
        "model": ResponseResource,
        "description": (
            "A completed response.\n\n"
            "With `stream: false` the body is a single `ResponseResource`: every field the spec "
            "marks required is present, nullable ones included.\n\n"
            "With `stream: true` the body is `text/event-stream` instead. Swagger UI cannot "
            "render a live stream, so use `curl -N` for that; the example below is a real trace, "
            "abridged."
        ),
        "content": {
            "text/event-stream": {
                "schema": {"type": "string"},
                "example": SSE_EXAMPLE,
            }
        },
    },
    404: {
        "description": (
            "`previous_response_id` was supplied. Nothing is stored here, so no id is ever known."
        ),
        "content": {"application/json": {"example": NO_CONTINUATION_EXAMPLE}},
    },
}

UPLOAD_DOCS: ResponseDocs = {
    202: {"description": "Accepted and queued. Poll `GET /documents/{document_id}` for progress."},
    400: {
        "description": "Empty upload.",
        "content": {"application/json": {"example": {"detail": "empty upload"}}},
    },
    413: {
        "description": "Larger than `API_MAX_UPLOAD_BYTES` (50 MB by default).",
        "content": {
            "application/json": {"example": {"detail": "73400320 bytes exceeds limit of 52428800"}}
        },
    },
}

NOT_FOUND_DOCS: ResponseDocs = {
    404: {
        "description": "No document with that id.",
        "content": {"application/json": {"example": {"detail": "document not found"}}},
    }
}
