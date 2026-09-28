"""Tools the client implements, offered to the model before the graph runs.

A function tool is externally hosted: the implementation belongs to the caller, so the only
correct thing to do with a call is hand it back and end the turn. That cannot happen inside the
graph, whose tool node would try to execute it.
"""

import json
from typing import TypeAlias

from core.chat import get_chat_model
from core.config import ModelSize
from langchain_core.messages import AnyMessage

from api.schemas.responses import CreateResponseBody, FunctionCallItem, FunctionToolParam

ToolCallDict: TypeAlias = dict[str, str | dict[str, object] | None]


def _tool_schema(tool: FunctionToolParam) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }


def client_tool_calls(
    body: CreateResponseBody, size: ModelSize, messages: list[AnyMessage]
) -> list[FunctionCallItem]:
    """Return the calls the model wants to make against the client's own tools, if any."""
    if not body.tools or body.tool_choice == "none":
        return []
    model = get_chat_model(size).bind_tools([_tool_schema(t) for t in body.tools])
    reply = model.invoke(messages)
    calls: list[ToolCallDict] = getattr(reply, "tool_calls", [])
    declared = {tool.name for tool in body.tools}
    items: list[FunctionCallItem] = []
    for call in calls:
        name = str(call["name"])
        if name not in declared:
            continue
        # The provider's own call id is reused when there is one, so a client that round-trips a
        # `function_call_output` matches it against what its model actually emitted.
        call_id = str(call.get("id") or "")
        item = FunctionCallItem(name=name, arguments=json.dumps(call.get("args") or {}))
        if call_id:
            item.call_id = call_id
        items.append(item)
    return items
