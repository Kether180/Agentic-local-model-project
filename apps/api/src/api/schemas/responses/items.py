"""Output items: what the model produced, in the shapes the spec names.

Two of the five spec item types are implemented — `message` and `function_call`. `reasoning`,
`function_call_output` and `compaction` are not produced by this server.
"""

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from api.schemas.responses.common import ItemStatus, new_id


class UrlCitation(BaseModel):
    """The spec's only annotation variant, and therefore how our references travel.

    `url` points at this API's own document route rather than the open web — the spec constrains
    the shape, not the scheme.
    """

    type: Literal["url_citation"] = "url_citation"
    url: str
    title: str
    start_index: int
    end_index: int


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[UrlCitation] = []


class MessageItem(BaseModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: new_id("msg"))
    status: ItemStatus = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[OutputText] = []


class FunctionCallItem(BaseModel):
    """A client tool the model chose to call. Terminal: control returns to the client."""

    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory=lambda: new_id("fc"))
    call_id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: str
    status: ItemStatus = "completed"


OutputItem: TypeAlias = Annotated[MessageItem | FunctionCallItem, Field(discriminator="type")]
