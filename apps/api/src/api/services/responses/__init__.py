"""Running one turn and emitting it in the shape the spec wants.

| | |
|---|---|
| `translate` | request body to LangChain messages, and `model` to a configured size |
| `client_tools` | tools the caller implements, offered before the graph runs |
| `resource` | assembling the response resource, completed or failed |

This module holds the two entry points and the SSE framing. The graph itself stays a graph: it
takes messages and returns an answer, and knows nothing about response ids, sequence numbers or
event names.
"""

import json
from collections.abc import Iterator

from core.config import ModelSize
from core.embedding import Embedder
from langchain_core.messages import AIMessageChunk
from sqlalchemy.orm import Session

from api.crud.chunk import ChunkHit
from api.schemas.responses import (
    AnnotationAddedEvent,
    AnyStreamEvent,
    ContentPartEvent,
    CreateResponseBody,
    MessageItem,
    OutputItem,
    OutputItemEvent,
    OutputText,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseResource,
    ResponseSnapshotEvent,
    UrlCitation,
)
from api.services.agent import ChatContext, ChatState, get_graph, parse_markers
from api.services.responses.client_tools import client_tool_calls
from api.services.responses.resource import answer_item, build_resource, completed, failed
from api.services.responses.translate import IMAGE_PLACEHOLDER, resolve_model, to_langchain

__all__ = [
    "IMAGE_PLACEHOLDER",
    "client_tool_calls",
    "create",
    "frame",
    "resolve_model",
    "stream",
    "to_langchain",
]

DONE = "data: [DONE]\n\n"
"""The spec terminates an event stream with this literal."""


def run_graph(
    body: CreateResponseBody, size: ModelSize, context: ChatContext
) -> tuple[str, list[ChunkHit]]:
    """Run the graph to completion and return the answer plus the chunks it saw."""
    messages = to_langchain(body.messages(), body.instructions)
    state = get_graph(size).invoke(ChatState(messages=messages), context=context)
    return state["messages"][-1].text, list(state.get("hits") or [])


def create(body: CreateResponseBody, session: Session, embedder: Embedder) -> ResponseResource:
    """The non-streaming path."""
    size, model_id = resolve_model(body.model)
    resource = build_resource(body, model_id, [], status="in_progress")
    try:
        messages = to_langchain(body.messages(), body.instructions)
        calls = client_tool_calls(body, size, messages)
        if calls:
            output: list[OutputItem] = list(calls)
        else:
            text, hits = run_graph(body, size, ChatContext(session=session, embedder=embedder))
            output = [answer_item(text, hits)]
    except Exception as err:  # noqa: BLE001 — every failure is a `failed` response; `failed` logs it
        return failed(resource, err)
    return completed(resource, output)


def frame(event: AnyStreamEvent, sequence: int) -> str:
    """One SSE frame. The `event:` name mirrors the payload `type`, which the spec requires."""
    event.sequence_number = sequence
    payload = json.dumps(event.serialize(), separators=(",", ":"))
    return f"event: {event.type}\ndata: {payload}\n\n"


def _answer_deltas(
    body: CreateResponseBody, size: ModelSize, context: ChatContext
) -> Iterator[tuple[str, list[ChunkHit]]]:
    """Yield `(delta, hits)` as the graph runs; the last yield carries the final hits.

    `subgraphs=True` is load-bearing. The answer is generated inside the compiled `create_agent`
    graph mounted as the `agent` node, and LangGraph does not surface a subgraph's token stream
    unless asked — without it this yields nothing at all and every answer streams as empty.

    Only chunks from the agent are forwarded. The filter-detection node also calls a model, and
    its structured-output tokens are machinery, not an answer.
    """
    messages = to_langchain(body.messages(), body.instructions)
    hits: list[ChunkHit] = []
    stream_ = get_graph(size).stream(
        ChatState(messages=messages),
        context=context,
        stream_mode=["messages", "values"],
        subgraphs=True,
    )
    for _namespace, mode, payload in stream_:
        if mode == "values":
            if isinstance(payload, dict) and payload.get("hits"):
                hits = list(payload["hits"])
        elif mode == "messages" and isinstance(payload, tuple):
            chunk, meta = payload
            if isinstance(meta, dict) and meta.get("langgraph_node") == "detect_filters":
                continue
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                yield chunk.text, hits
    yield "", hits


def stream(body: CreateResponseBody, session: Session, embedder: Embedder) -> Iterator[str]:
    """The streaming path, in the item lifecycle order the spec mandates.

    `response.created` and `response.in_progress` carry an empty snapshot; the terminal
    `response.completed` carries the assembled one, which is the snapshot the acceptance suite
    validates.
    """
    size, model_id = resolve_model(body.model)
    sequence = 0
    # One resource, updated as the turn progresses. Rebuilding it per event would mint a new `id`
    # and `created_at` each time, and a client correlating `response.created` with
    # `response.completed` would see two unrelated responses.
    opening = build_resource(body, model_id, [], status="in_progress")

    def emit(event: AnyStreamEvent) -> str:
        nonlocal sequence
        out = frame(event, sequence)
        sequence += 1
        return out

    yield emit(ResponseSnapshotEvent(type="response.created", response=opening))
    yield emit(ResponseSnapshotEvent(type="response.in_progress", response=opening))

    try:
        messages = to_langchain(body.messages(), body.instructions)
        calls = client_tool_calls(body, size, messages)
    except Exception as err:  # noqa: BLE001 — see `resource.failed`
        yield emit(ResponseSnapshotEvent(type="response.failed", response=failed(opening, err)))
        yield DONE
        return

    if calls:
        for index, call in enumerate(calls):
            yield emit(
                OutputItemEvent(type="response.output_item.added", output_index=index, item=call)
            )
            yield emit(
                OutputItemEvent(type="response.output_item.done", output_index=index, item=call)
            )
        yield emit(
            ResponseSnapshotEvent(
                type="response.completed", response=completed(opening, list(calls))
            )
        )
        yield DONE
        return

    item = MessageItem(status="in_progress")
    yield emit(OutputItemEvent(type="response.output_item.added", output_index=0, item=item))
    yield emit(
        ContentPartEvent(
            type="response.content_part.added",
            item_id=item.id,
            output_index=0,
            content_index=0,
            part=OutputText(text=""),
        )
    )

    context = ChatContext(session=session, embedder=embedder)
    text = ""
    hits: list[ChunkHit] = []
    try:
        for delta, current_hits in _answer_deltas(body, size, context):
            hits = current_hits
            if not delta:
                continue
            text += delta
            yield emit(
                OutputTextDeltaEvent(item_id=item.id, output_index=0, content_index=0, delta=delta)
            )
    except Exception as err:  # noqa: BLE001 — see `resource.failed`
        # The spec requires `response.failed` to follow an error encountered mid-stream. The item
        # opened above is simply abandoned — there is no partial-item close in the spec.
        yield emit(ResponseSnapshotEvent(type="response.failed", response=failed(opening, err)))
        yield DONE
        return

    citations: list[UrlCitation] = parse_markers(text, hits)
    for index, citation in enumerate(citations):
        yield emit(
            AnnotationAddedEvent(
                item_id=item.id,
                output_index=0,
                content_index=0,
                annotation_index=index,
                annotation=citation,
            )
        )

    part = OutputText(text=text, annotations=citations)
    yield emit(OutputTextDoneEvent(item_id=item.id, output_index=0, content_index=0, text=text))
    yield emit(
        ContentPartEvent(
            type="response.content_part.done",
            item_id=item.id,
            output_index=0,
            content_index=0,
            part=part,
        )
    )
    done = MessageItem(id=item.id, status="completed", content=[part])
    yield emit(OutputItemEvent(type="response.output_item.done", output_index=0, item=done))
    yield emit(
        ResponseSnapshotEvent(type="response.completed", response=completed(opening, [done]))
    )
    yield DONE
