"""The chat graph."""

from functools import lru_cache
from typing import TypedDict
from uuid import uuid4

from core.chat import get_chat_model
from core.config import ModelSize
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from api.crud.chunk import ChunkHit
from api.services.agent import filters as filter_service
from api.services.agent.filters import AutoFilter
from api.services.agent.state import ChatContext, ChatState
from api.services.agent.tools import format_hits, run_search, search_documents

SYSTEM_PROMPT = """Answer questions about the ingested document corpus.

The search results for the question are already above, labelled [S1], [S2] and so on. Answer \
only from them, ending every sentence you took from a result with that result's label, like [S2]. \
Call `search_documents` only to search again with different words.

Say so plainly if the results do not contain the answer. Answer in short prose."""


class FilterUpdate(TypedDict):
    """What `_detect_filters` writes. A node returns a partial update, not a whole `ChatState`,
    and `ChatState` requires `messages` — so the update gets its own type rather than a lie.
    """

    filters: AutoFilter


class SearchUpdate(TypedDict):
    """What `_search` writes: the hits, and the results as a completed tool call."""

    hits: list[ChunkHit]
    messages: list[AnyMessage]


def _question(state: ChatState) -> str:
    return next((m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


def _detect_filters(state: ChatState, runtime: Runtime[ChatContext]) -> FilterUpdate:
    """Infer metadata filters from the latest question. Writes state, appends no message."""
    del runtime  # required by the node signature; this node only needs the question
    return {"filters": filter_service.infer(_question(state))}


def _search(state: ChatState, runtime: Runtime[ChatContext]) -> SearchUpdate:
    """Search the question before the model speaks, so every answer is grounded in results.

    Left to the model, a small one skips the search and cites labels that point at nothing. The
    results arrive as a completed `search_documents` call, the shape every provider accepts; the
    model can still call the tool again to search with other words."""

    question = _question(state)
    hits, filters = run_search(question, state.get("filters") or AutoFilter(), runtime.context)
    call_id = f"search_{uuid4().hex}"
    call = ToolCall(name=search_documents.name, args={"query": question}, id=call_id)

    return {
        "hits": hits,
        "messages": [
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=format_hits(hits, 0, filters), tool_call_id=call_id),
        ],
    }


def build_graph(
    size: ModelSize = "small",
) -> StateGraph[ChatState, ChatContext, ChatState, ChatState]:
    """Wire the graph. Uncompiled, so tests can compile it against a stub."""
    graph = StateGraph(ChatState, context_schema=ChatContext)
    agent = create_agent(  # wraps the model in a loop , model replies and if it asks for a tool, langraph runs the tool and give the result
        model=get_chat_model(size), # core/chat.py -> Ollama, qwen3.5
        tools=[search_documents], # tools.py the model MAY call it
        system_prompt=SYSTEM_PROMPT,
        state_schema=ChatState,
        context_schema=ChatContext,
    )
    graph.add_node("detect_filters", _detect_filters)
    graph.add_node("search", _search)
    graph.add_node("agent", agent)
    graph.add_edge(START, "detect_filters")
    graph.add_edge("detect_filters", "search")
    graph.add_edge("search", "agent")
    graph.add_edge("agent", END)

    return graph


@lru_cache(maxsize=len(ModelSize.__args__))
def get_graph(
    size: ModelSize = "small",
) -> CompiledStateGraph[ChatState, ChatContext, ChatState, ChatState]:
    """One compiled graph per model size, built on first use."""
    return build_graph(size).compile()
