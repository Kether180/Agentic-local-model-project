"""Graph state and request-scoped context."""

import operator
from dataclasses import dataclass
from typing import Annotated, NotRequired

from core.embedding import Embedder
from langchain.agents import AgentState
from sqlalchemy.orm import Session

from api.crud.chunk import ChunkHit
from api.services.agent.filters import AutoFilter


@dataclass
class ChatContext:
    """Injected per request via LangGraph's `context_schema`, which is how the search tool reaches
    postgres without a module-level session."""

    session: Session
    embedder: Embedder
    top_k: int = 10


class ChatState(AgentState[None]):  # memory of the chat , langchain standard's memory
    """`AgentState` is extended rather than replaced because `create_agent` does not merge its own
    fields into a schema it is given — an agent built on a bare schema loses `jump_to` and every
    middleware that depends on it.

    filters: what `detect_filters` inferred; read by the search tool at call time.
    hits: every chunk the search tool returned this turn, in the order it numbered them, so
        `[S3]` in the answer can be resolved back to a chunk.
    """

    filters: NotRequired[AutoFilter]
    hits: NotRequired[Annotated[list[ChunkHit], operator.add]]
