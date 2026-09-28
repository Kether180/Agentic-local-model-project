"""OpenResponses wire types, transcribed from https://www.openresponses.org/openapi/openapi.json.

| | |
|---|---|
| `common` | shared literals and id generation |
| `requests` | the `POST /responses` body — this is what `/docs` renders |
| `items` | output items: messages, citations, function calls |
| `resource` | the response resource every reply is validated against |
| `events` | the closed set of SSE frames |
"""

from api.schemas.responses.common import ItemStatus, MessageRole, ResponseStatus, new_id
from api.schemas.responses.events import (
    AnnotationAddedEvent,
    AnyStreamEvent,
    ContentPartEvent,
    OutputItemEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseSnapshotEvent,
    StreamEvent,
)
from api.schemas.responses.items import (
    FunctionCallItem,
    MessageItem,
    OutputItem,
    OutputText,
    UrlCitation,
)
from api.schemas.responses.requests import (
    ContentPart,
    CreateResponseBody,
    FunctionToolParam,
    InputMessage,
    ToolChoice,
)
from api.schemas.responses.resource import (
    IncompleteDetails,
    ResponseError,
    ResponseResource,
    TextField,
    Usage,
)

__all__ = [
    "AnnotationAddedEvent",
    "AnyStreamEvent",
    "ContentPart",
    "ContentPartEvent",
    "CreateResponseBody",
    "FunctionCallItem",
    "FunctionToolParam",
    "IncompleteDetails",
    "InputMessage",
    "ItemStatus",
    "MessageItem",
    "MessageRole",
    "OutputItem",
    "OutputItemEvent",
    "OutputText",
    "OutputTextDeltaEvent",
    "OutputTextDoneEvent",
    "ResponseError",
    "ResponseResource",
    "ResponseSnapshotEvent",
    "ResponseStatus",
    "StreamEvent",
    "TextField",
    "ToolChoice",
    "UrlCitation",
    "Usage",
    "new_id",
]
