"""The two guards that keep autofiltering from making answers worse.

Both exist because of things a 0.8b model actually did against the sample corpus, not because of
things it might do: it invented `author_last_name='Schwartz' year=1958` for a question containing
neither, and the citation step it filters against left `authors` empty while putting the journal
name in `title`.
"""

from domain.pg import Chunk, Document
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from api.services.agent.filters import AutoFilter, _grounded


def sql(filters: AutoFilter) -> str:
    stmt = (
        select(Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(*filters.conditions())
    )
    return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def test_ungrounded_values_are_dropped() -> None:
    """Every filter is meant to be a literal lift from the question. This one was invented."""
    question = "What does the corpus say about amyloid beta aggregation?"

    kept = _grounded(AutoFilter(author_last_name="Schwartz", year=1958), question)

    assert kept == AutoFilter()


def test_grounded_values_survive() -> None:
    question = "What do papers by Chen from 2017 in Acta Pharmacologica Sinica say about tau?"
    detected = AutoFilter(author_last_name="Chen", year=2017, journal="Acta Pharmacologica Sinica")

    assert _grounded(detected, question) == detected


def test_grounding_ignores_case() -> None:
    assert _grounded(AutoFilter(author_last_name="CHEN"), "papers by chen") == AutoFilter(
        author_last_name="CHEN"
    )


def test_author_also_matches_the_filename() -> None:
    """`authors` is routinely empty — the citation step is a small model too."""
    rendered = sql(AutoFilter(author_last_name="Chen"))

    assert "authors" in rendered
    assert "document.filename" in rendered.lower()


def test_journal_also_matches_the_title() -> None:
    """Observed on the sample paper: the journal name was extracted into `title`."""
    rendered = sql(AutoFilter(journal="Acta"))

    assert "'journal'" in rendered
    assert "'title'" in rendered


def test_no_filters_means_no_where_clause() -> None:
    assert AutoFilter().is_empty
    assert AutoFilter().conditions() == []
    assert "WHERE" not in sql(AutoFilter())
