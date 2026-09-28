"""The agent's one tool."""

import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from api.crud import chunk as chunk_crud
from api.crud.chunk import ChunkHit
from api.services.agent.filters import AutoFilter
from api.services.agent.references import attach_sources
from api.services.agent.state import ChatContext, ChatState

logger = logging.getLogger(__name__)


def format_hits(hits: list[ChunkHit], offset: int, filters: AutoFilter) -> str:
    """Number each chunk so the model has a token to cite it with.

    `offset` is how many chunks earlier calls in this same turn already numbered, which keeps
    `[S1]` meaning one thing across a turn where the model searches twice.

    An active filter is reported here rather than in the message history: the model needs it to
    explain a thin result set, and this is the only place it is unambiguously about the results.
    """
    scope = "" if filters.is_empty else f"{filters.describe()}\n"
    if not hits:
        return f"<results>\n{scope}no matching chunks\n</results>"
    blocks = [
        f"[S{offset + i + 1}] {hit.filename} p.{hit.page} (score {hit.score:.3f})\n{hit.text}"
        for i, hit in enumerate(hits)
    ]
    labels = ", ".join(f"[S{offset + i + 1}]" for i in range(len(hits)))
    return (
        f"<results>\n{scope}" + "\n\n".join(blocks) + "\n</results>\n"
        # Repeated after the results, not only in the system prompt: a small model follows the
        # most recent instruction, and the system prompt is a long way back by now.
        f"Cite these as {labels}. Every sentence you take from them must end with its label."
    )


def run_search(
    query: str, filters: AutoFilter, context: ChatContext
) -> tuple[list[ChunkHit], AutoFilter]:
    """Embed `query`, search, and resolve each hit's citation marks. Returns the hits and the
    filters actually applied, which differ when an empty filtered search was widened."""
    vector = context.embedder.embed(query)
    hits = chunk_crud.search(context.session, vector, context.top_k, filters=filters)
    if not hits and not filters.is_empty:
        # A filter that matches nothing is worse than no filter: the corpus looks empty and the
        # model reports there is nothing to say. Inferred filters are a guess about what the user
        # meant, and the citation metadata they match against is itself model-extracted, so either
        # side can be wrong. Widening beats answering "no results" from a corpus that has them.
        logger.info("filters %s matched nothing; retrying unfiltered", filters)
        filters = AutoFilter()
        hits = chunk_crud.search(context.session, vector, context.top_k)
    # Second order references: resolve each hit's citation marks against its document's
    # references section, so `[Sn]` can point at the original source rather than the chunk.
    pages = chunk_crud.page_texts(context.session, {hit.document_id for hit in hits})
    return attach_sources(hits, pages), filters


@tool
def search_documents(query: str, runtime: ToolRuntime[ChatContext, ChatState]) -> Command[str]:
    """Semantic search over the ingested document corpus.

    Results are chunks of documents, not whole documents: one document contributes many chunks, so
    a chunk count never answers "how many documents".

    Each result is labelled `[S1]`, `[S2]`, and so on. Cite the label inline, immediately after the
    sentence it supports, for every claim you take from a result.

    Any metadata filters detected from the question are applied automatically.
    """
    state_hits: list[ChunkHit] = runtime.state.get("hits") or []
    filters: AutoFilter = runtime.state.get("filters") or AutoFilter()
    hits, filters = run_search(query, filters, runtime.context)
    return Command[str](
        update={
            "hits": hits,
            "messages": [
                ToolMessage(
                    content=format_hits(hits, offset=len(state_hits), filters=filters),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
