"""The response resource.

Every field the spec marks required is declared here with a default, including the nullable ones,
and `serialize()` dumps the model whole. The compliance runner parses this with a generated zod
schema: a field omitted because it was null fails that parse exactly as hard as a wrong value.
"""

import time
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from api.schemas.responses.common import ResponseStatus, ServiceTier, Truncation, new_id
from api.schemas.responses.items import OutputItem
from api.schemas.responses.requests import FunctionToolParam, ToolChoice


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    """Required by the spec, reported as zeros — this server does not count tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails = InputTokensDetails()
    output_tokens_details: OutputTokensDetails = OutputTokensDetails()


class TextFormat(BaseModel):
    type: Literal["text"] = "text"


class TextField(BaseModel):
    format: TextFormat = TextFormat()


class ReasoningField(BaseModel):
    effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    summary: Literal["concise", "detailed", "auto"] | None = None


class IncompleteDetails(BaseModel):
    reason: str


class ResponseError(BaseModel):
    code: str
    message: str


class ResponseResource(BaseModel):
    id: str = Field(default_factory=lambda: new_id("resp"))
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    completed_at: int | None = None
    status: ResponseStatus = "in_progress"
    incomplete_details: IncompleteDetails | None = None
    model: str = ""
    previous_response_id: str | None = None
    instructions: str | None = None
    output: list[OutputItem] = []
    error: ResponseError | None = None
    tools: list[FunctionToolParam] = []
    tool_choice: ToolChoice = "auto"
    truncation: Truncation = "disabled"
    parallel_tool_calls: bool = True
    text: TextField = TextField()
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = 0
    temperature: float = 1.0
    reasoning: ReasoningField = ReasoningField()
    usage: Usage = Usage()
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool = True
    background: bool = False
    service_tier: ServiceTier = "default"
    metadata: dict[str, str] = {}
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None

    def serialize(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")
