"""Assembling the response resource: the outbound half of the translation."""

import logging
import time

from api.crud.chunk import ChunkHit
from api.schemas.responses import (
    CreateResponseBody,
    MessageItem,
    OutputItem,
    OutputText,
    ResponseError,
    ResponseResource,
)
from api.services.agent import parse_markers

log = logging.getLogger(__name__)


def answer_item(text: str, hits: list[ChunkHit]) -> MessageItem:
    """The assistant's message, with every `[Sn]` marker resolved to a citation."""
    return MessageItem(content=[OutputText(text=text, annotations=parse_markers(text, hits))])


def build_resource(
    body: CreateResponseBody, model_id: str, output: list[OutputItem], status: str = "completed"
) -> ResponseResource:
    """A response resource carrying the request's echoed settings.

    Echoing is not decoration: the spec requires these fields, and a client reads them to learn
    what the server actually applied.
    """
    return ResponseResource(
        status=status,  # pyright: ignore[reportArgumentType]  # callers pass spec statuses
        completed_at=int(time.time()) if status in ("completed", "failed") else None,
        model=model_id,
        instructions=body.instructions,
        output=output,
        tools=body.tools,
        tool_choice=body.tool_choice,
        truncation=body.truncation,
        parallel_tool_calls=body.parallel_tool_calls,
        top_p=body.top_p,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        top_logprobs=body.top_logprobs,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
        max_tool_calls=body.max_tool_calls,
        store=body.store,
        background=body.background,
        service_tier=body.service_tier,
        metadata=body.metadata,
        safety_identifier=body.safety_identifier,
        prompt_cache_key=body.prompt_cache_key,
        previous_response_id=body.previous_response_id,
    )


def completed(resource: ResponseResource, output: list[OutputItem]) -> ResponseResource:
    """Close out a turn.

    An update of the opening resource rather than a fresh one, so `id` and `created_at` survive —
    a client correlating `response.created` with `response.completed` must see one response, not
    two.
    """
    return resource.model_copy(
        update={"status": "completed", "completed_at": int(time.time()), "output": output}
    )


def failed(resource: ResponseResource, err: Exception) -> ResponseResource:
    """Turn an exception into a response the client can actually read.

    A model or database failure is a `failed` response with an `error` object — not a 500 with a
    traceback. The spec models failure as a state of the response, and a client that gets HTML
    where it expects a response resource cannot even tell you what went wrong.
    """
    log.exception("response %s failed", resource.id)
    return resource.model_copy(
        update={
            "status": "failed",
            "completed_at": int(time.time()),
            "error": ResponseError(code="model_error", message=str(err) or type(err).__name__),
        }
    )
