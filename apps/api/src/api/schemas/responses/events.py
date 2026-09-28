"""SSE frames.

This list is closed, and that is the point. The compliance runner validates every frame against a
union of exactly the spec's event types and reports anything else as an error, so an event of our
own invention — a `references` channel, say — would subtract from compliance rather than add to
it. References travel as `url_citation` annotations on `response.output_text.annotation.added`
because that is the spec's own shape for them.
"""

from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue

from api.schemas.responses.items import OutputItem, OutputText, UrlCitation
from api.schemas.responses.resource import ResponseResource


class StreamEvent(BaseModel):
    """Base for every frame. `sequence_number` is assigned by the emitter, not here.

    `type` is declared on each subclass rather than here so it can stay a `Literal` — narrowing an
    inherited `str` would be an incompatible override. `AnyStreamEvent` is what callers annotate
    against, and every member of it has a `type`.
    """

    sequence_number: int = 0

    def serialize(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


class ResponseSnapshotEvent(StreamEvent):
    """`response.created`, `.in_progress`, `.completed`, `.failed` — all carry a full snapshot."""

    type: Literal[
        "response.created", "response.in_progress", "response.completed", "response.failed"
    ]
    response: ResponseResource


class OutputItemEvent(StreamEvent):
    type: Literal["response.output_item.added", "response.output_item.done"]
    output_index: int
    item: OutputItem


class ContentPartEvent(StreamEvent):
    type: Literal["response.content_part.added", "response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: OutputText


class OutputTextDeltaEvent(StreamEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str


class OutputTextDoneEvent(StreamEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str


class AnnotationAddedEvent(StreamEvent):
    """How a reference reaches the client mid-stream."""

    type: Literal["response.output_text.annotation.added"] = "response.output_text.annotation.added"
    item_id: str
    output_index: int
    content_index: int
    annotation_index: int
    annotation: UrlCitation


AnyStreamEvent: TypeAlias = (
    ResponseSnapshotEvent
    | OutputItemEvent
    | ContentPartEvent
    | OutputTextDeltaEvent
    | OutputTextDoneEvent
    | AnnotationAddedEvent
)
"""Every frame this server emits."""
