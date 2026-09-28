"""The `POST /responses` request body.

Field descriptions and examples here are what `/docs` renders, so they are written for someone
reading the Swagger page rather than the source: what the field does *on this server*, not what
the spec says in general. Where the two differ — `model`, `previous_response_id` — the field says
so.

Input models allow extra keys on purpose. The "Assistant Message Phase" acceptance test sends a
`phase` field that is not in the schema and expects the server to take it anyway.
"""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from api.schemas.responses.common import MessageRole, ServiceTier, Truncation

SIMPLE_EXAMPLE: dict[str, JsonValue] = {
    "model": "small",
    "input": "What does the corpus say about amyloid beta aggregation?",
}

STREAMING_EXAMPLE: dict[str, JsonValue] = {
    "model": "small",
    "stream": True,
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": "Summarise what the corpus says about tau phosphorylation.",
        }
    ],
}

CONVERSATION_EXAMPLE: dict[str, JsonValue] = {
    "model": "medium",
    "instructions": "Answer in one sentence.",
    "input": [
        {"type": "message", "role": "user", "content": "Which paper is in the corpus?"},
        {"type": "message", "role": "assistant", "content": "A review of amyloid beta."},
        {"type": "message", "role": "user", "content": "Who wrote it?"},
    ],
}

TOOL_EXAMPLE: dict[str, JsonValue] = {
    "model": "small",
    "input": "What is the weather in San Francisco?",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
}


class ContentPart(BaseModel):
    """One part of a multi-part message.

    Deliberately one permissive model rather than a discriminated union over
    `input_text`/`input_image`/`input_file`/`output_text`: the only part we read is the text. A
    union here would be five classes to reject nothing.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        description="`input_text`, `input_image`, or `input_file`.", examples=["input_text"]
    )
    text: str | None = Field(default=None, examples=["What is in the corpus?"])
    image_url: str | None = Field(
        default=None,
        description=(
            "Accepted, but not sent to the model: this server answers from a text corpus and the "
            "default chat model has no vision. The model is told an image was attached."
        ),
    )
    detail: Literal["low", "high", "auto"] | None = None


class InputMessage(BaseModel):
    """One message item in the request `input` array."""

    model_config = ConfigDict(extra="allow")

    type: Literal["message"] = "message"
    role: MessageRole = Field(
        description="`developer` is treated as `system`; the providers here do not distinguish them.",
        examples=["user"],
    )
    content: str | list[ContentPart] = Field(
        description="Plain text, or a list of content parts for multi-part input.",
        examples=["What does the corpus say about amyloid beta aggregation?"],
    )
    id: str | None = None
    status: str | None = None


class FunctionToolParam(BaseModel):
    """A function tool the *client* implements.

    Calls to it are returned as a `function_call` output item and the turn ends there — the
    implementation is yours, so control goes back to you rather than the model looping on it.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: str = Field(examples=["get_weather"])
    description: str | None = Field(
        default=None, examples=["Get the current weather for a location"]
    )
    parameters: dict[str, JsonValue] | None = Field(
        default=None,
        description="JSON Schema for the arguments.",
        examples=[{"type": "object", "properties": {"location": {"type": "string"}}}],
    )
    strict: bool | None = None


class SpecificFunction(BaseModel):
    name: str


class SpecificToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: SpecificFunction | None = None
    name: str | None = None


ToolChoice: TypeAlias = Literal["none", "auto", "required"] | SpecificToolChoice


class CreateResponseBody(BaseModel):
    """Ask a question about the ingested document corpus.

    Only a handful of these fields change what the server does; the rest are accepted so a
    compliant client is never rejected, and echoed back on the response resource so it can see
    what was applied.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [SIMPLE_EXAMPLE, STREAMING_EXAMPLE, CONVERSATION_EXAMPLE, TOOL_EXAMPLE]
        },
    )

    model: str = Field(
        description=(
            "`small`, `medium` or `large`. These map to the models configured for the active "
            "provider; any other value falls back to `small`. The response echoes the provider's "
            "own model id, e.g. `qwen3.5:0.8b`."
        ),
        examples=["small"],
    )
    input: str | list[InputMessage] = Field(
        default="",
        description=(
            "The conversation. A bare string is shorthand for a single user message. Send the "
            "whole history here — this server stores nothing between requests."
        ),
        examples=["What does the corpus say about amyloid beta aggregation?"],
    )
    instructions: str | None = Field(
        default=None,
        description="Prepended as a system message, in addition to the agent's own prompt.",
        examples=["Answer in one sentence."],
    )
    tools: list[FunctionToolParam] = Field(
        default=[],
        description=(
            "Function tools you implement. The corpus search tool is always available and is not "
            "listed here."
        ),
    )
    tool_choice: ToolChoice = Field(default="auto", examples=["auto"])
    stream: bool = Field(
        default=False,
        description=(
            "Stream the answer as `text/event-stream`. Citations arrive as "
            "`response.output_text.annotation.added` events while the answer is still being "
            "written."
        ),
        examples=[False],
    )
    store: bool = Field(
        default=True, description="Echoed, but nothing is stored. See `previous_response_id`."
    )
    background: bool = Field(default=False, description="Echoed; not supported.")
    previous_response_id: str | None = Field(
        default=None,
        description=(
            "Not supported — responses are never stored, so any value returns 404 "
            "`previous_response_not_found`. Put the history in `input` instead."
        ),
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, examples=[1.0])
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, examples=[1.0])
    presence_penalty: float = Field(default=0.0, description="Echoed; not applied.")
    frequency_penalty: float = Field(default=0.0, description="Echoed; not applied.")
    top_logprobs: int = Field(default=0, description="Echoed; log probabilities are not returned.")
    max_output_tokens: int | None = Field(default=None, description="Echoed; not applied.")
    max_tool_calls: int | None = Field(default=None, description="Echoed; not applied.")
    truncation: Truncation = Field(default="disabled", description="Echoed; not applied.")
    parallel_tool_calls: bool = Field(default=True, description="Echoed; not applied.")
    service_tier: ServiceTier = Field(default="default", description="Echoed; one tier exists.")
    metadata: dict[str, str] = Field(
        default={},
        description="Opaque key/value pairs, returned untouched on the response.",
        examples=[{"trace": "demo-1"}],
    )
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None

    def messages(self) -> list[InputMessage]:
        """Normalize the two input shapes into one. A bare string is a user message."""
        if isinstance(self.input, str):
            return [InputMessage(role="user", content=self.input)] if self.input else []
        return self.input
