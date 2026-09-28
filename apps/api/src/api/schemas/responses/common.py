"""Shared vocabulary for the OpenResponses wire types.

Transcribed from https://www.openresponses.org/openapi/openapi.json.
"""

import uuid
from typing import Literal, TypeAlias

MessageRole: TypeAlias = Literal["user", "assistant", "system", "developer"]
ItemStatus: TypeAlias = Literal["in_progress", "completed", "incomplete"]
ResponseStatus: TypeAlias = Literal["queued", "in_progress", "completed", "failed", "incomplete"]
ServiceTier: TypeAlias = Literal["auto", "default", "flex", "priority"]
Truncation: TypeAlias = Literal["auto", "disabled"]


def new_id(prefix: str) -> str:
    """Opaque, prefixed identifier — `resp_…`, `msg_…`, `fc_…`, `call_…`."""
    return f"{prefix}_{uuid.uuid4().hex}"
