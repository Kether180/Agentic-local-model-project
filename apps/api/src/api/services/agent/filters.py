"""Autofilter detection.

The three fields are not arbitrary: they are the `CitationMetadata` fields the citation step
populates densely enough to be worth filtering on. Filtering on a field the extractor usually
leaves null would silently return nothing.
"""

import json
import logging

from core.chat import get_chat_model
from domain.pg import Document
from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, cast, literal, or_
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)

_INSTRUCTIONS = """Extract literal metadata filters from the user's question about a document \
corpus.

Only fill a field when the question names it explicitly. "papers by Chen" gives an author; \
"what causes plaque formation" gives nothing at all. Leave every field null when in doubt — a \
wrong filter hides the answer entirely, where no filter merely searches wider."""


class AutoFilter(BaseModel):
    """Metadata constraints inferred from the question. All-null means "search everything"."""

    author_last_name: str | None = Field(
        default=None, description="Surname of an author the question names, e.g. 'Chen'."
    )
    year: int | None = Field(default=None, description="Four-digit publication year.")
    journal: str | None = Field(
        default=None, description="Journal or publication name, matched as a substring."
    )

    @property
    def is_empty(self) -> bool:
        return self.author_last_name is None and self.year is None and self.journal is None

    def conditions(self) -> list[ColumnElement[bool]]:
        """SQL predicates over `document.citation`.

        The column is a `PydanticJSONB`, so it is cast back to plain `JSONB` before indexing —
        the type decorator carries the pydantic model, not the JSONB comparator operators.

        Each predicate is widened past the field it nominally targets, because the citation step
        extracts with a small model and its output is not tidy. On the sample paper it produced
        `title="Acta Pharmacologica Sinica"` — the journal, in the title field — an empty `authors`
        list, and no date at all. A strict `authors @> [{"last_name": "Chen"}]` matches nothing
        there even though the document is plainly Chen's, so author also checks the filename and
        journal also checks the title.
        """
        citation = cast(Document.citation, JSONB)
        found: list[ColumnElement[bool]] = []
        if self.author_last_name is not None:
            # `@>` containment: true when any element of the authors array has this surname.
            probe = json.dumps([{"last_name": self.author_last_name}])
            found.append(
                or_(
                    citation["authors"].op("@>")(cast(literal(probe), JSONB)),
                    Document.filename.ilike(f"%{self.author_last_name}%"),
                )
            )
        if self.year is not None:
            found.append(citation[("publication_date", "year")].astext == str(self.year))
        if self.journal is not None:
            found.append(
                or_(
                    citation["journal"].astext.ilike(f"%{self.journal}%"),
                    citation["title"].astext.ilike(f"%{self.journal}%"),
                )
            )
        return found

    def describe(self) -> str:
        """One line for the model, so it knows why its search was scoped."""
        if self.is_empty:
            return "No metadata filters are active; the whole corpus is searchable."
        parts = [
            f"{name}={value!r}"
            for name, value in (
                ("author", self.author_last_name),
                ("year", self.year),
                ("journal", self.journal),
            )
            if value is not None
        ]
        return (
            f"Active metadata filters (applied automatically to every search): {', '.join(parts)}."
        )


def _grounded(detected: AutoFilter, question: str) -> AutoFilter:
    """Drop any inferred value that does not literally appear in the question.

    A 0.8b model asked for structured output fills fields it has no basis for — "what does the
    corpus say about amyloid beta aggregation" came back with `year=2021`, which then hides every
    document. Every filter this produces is meant to be a literal lift from the question, so
    requiring it to appear there costs nothing real and removes the whole failure mode.

    A larger model needs this far less, but it is cheap enough to leave on for all of them.
    """
    haystack = question.casefold()
    kept = AutoFilter(
        author_last_name=detected.author_last_name
        if detected.author_last_name and detected.author_last_name.casefold() in haystack
        else None,
        year=detected.year if detected.year and str(detected.year) in haystack else None,
        journal=detected.journal
        if detected.journal and detected.journal.casefold() in haystack
        else None,
    )
    if kept != detected:
        logger.info("dropped ungrounded autofilters: %s -> %s", detected, kept)
    return kept


def infer(question: str) -> AutoFilter:
    """Ask the small model for filters. A failure here is not fatal — it just means no filters.

    Structured output on a 0.8b model is the flakiest call in the graph, and an unfiltered search
    is a worse answer, not a broken one.
    """
    if not question.strip():
        return AutoFilter()
    model = get_chat_model("small").with_structured_output(AutoFilter)
    try:
        result = model.invoke([("system", _INSTRUCTIONS), ("human", question)])
    except Exception:
        logger.exception("autofilter detection failed, searching unfiltered")
        return AutoFilter()
    if isinstance(result, AutoFilter):
        return _grounded(result, question)
    logger.warning(f"autofilter returned {type(result).__name__}, searching unfiltered")
    return AutoFilter()
