"""Request body in, LangChain messages out.

The only direction-of-travel this module knows is inbound. Nothing here builds a response.
"""

import logging
from typing import get_args

from core.config import ModelSize, settings
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from api.schemas.responses import ContentPart, InputMessage

log = logging.getLogger(__name__)

SIZES: tuple[ModelSize, ...] = get_args(ModelSize)

IMAGE_PLACEHOLDER = (
    "[an image was attached; this server answers from a text corpus and does not read images]"
)


def resolve_model(requested: str) -> tuple[ModelSize, str]:
    """Map the request's `model` onto a configured size, and report the id actually used.

    The spec's `model` is an opaque provider string, and ours are sizes. A request naming a size
    gets that size; anything else gets `small` rather than a 404, so a client pointed at this
    server with its own default model string still works. The id returned is the provider's —
    `qwen3.5:0.8b` on Ollama, the Bedrock model id on Bedrock — because that is what generated the
    answer, and echoing back the client's string would claim otherwise.
    """
    size: ModelSize = "small"
    if requested in SIZES:
        size = requested  # pyright: ignore[reportAssignmentType]  # membership narrows it
    elif requested:
        log.warning("unknown model %r, falling back to %r", requested, size)
    return size, settings.chat.chat_models[size]


def _content_to_text(parts: list[ContentPart]) -> str:
    """Flatten content parts to text.

    Image and file parts are accepted and then described rather than forwarded. This is a
    document-corpus Q&A service whose answers come from indexed text, and the default chat model
    is not a vision model — handing it an image is a hard 400 from Ollama ("Failed to load image
    or audio file"), which turns an answerable question into a failed response.

    Telling the model an image was present is the honest version of that: it can say it cannot see
    it, instead of silently answering as though the image had never been sent. Forwarding images
    for real belongs behind a capability flag on the chat provider, once one of them can use them.
    """
    pieces: list[str] = []
    for part in parts:
        if part.text:
            pieces.append(part.text)
        elif part.image_url is not None:
            pieces.append(IMAGE_PLACEHOLDER)
    return "\n".join(pieces)


def _message(role: str, content: str) -> AnyMessage:
    """`developer` maps to `system`: the spec distinguishes them, the providers here do not."""
    if role == "assistant":
        return AIMessage(content=content)
    if role in ("system", "developer"):
        return SystemMessage(content=content)
    return HumanMessage(content=content)


def to_langchain(messages: list[InputMessage], instructions: str | None) -> list[AnyMessage]:
    """Input items to LangChain messages.

    Unknown extra fields on an item (the acceptance suite sends `phase`) are simply not read.
    """
    out: list[AnyMessage] = []
    if instructions:
        out.append(SystemMessage(content=instructions))
    for message in messages:
        content = (
            message.content
            if isinstance(message.content, str)
            else _content_to_text(message.content)
        )
        out.append(_message(message.role, content))
    return out
