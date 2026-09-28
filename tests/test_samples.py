"""The reference parser against the real sample corpus, offline.

Follows the production path: PyMuPDF extracts the pages as the parse step does, the pipeline's
own chunker splits them, and the pages are rebuilt from the chunks as `crud.page_texts` does.
"""

# PyMuPDF ships no type information; the pipeline's parse step reports the same two warnings.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from pathlib import Path

import pymupdf
import pytest

from api.services.agent.references import find_marks, parse_reference_lists
from pipeline.steps.chunk import split

SAMPLES = Path(__file__).parent.parent / "samples"
REFERENCES_PER_PDF = {1: 50, 2: 24, 3: 24, 4: 34, 5: 135}


def chunks_of(pdf: Path) -> list[tuple[int, str]]:
    """(page number, chunk text) for every chunk, in document order."""
    with pymupdf.open(pdf) as doc:
        pages = [str(page.get_text()) for page in doc]
    return [(number, chunk) for number, text in enumerate(pages, start=1) for chunk in split(text)]


def rebuild_pages(chunks: list[tuple[int, str]]) -> dict[int, str]:
    pages: dict[int, str] = {}
    for page, chunk in chunks:
        pages[page] = f"{pages[page]}\n{chunk}" if page in pages else chunk
    return pages


@pytest.mark.parametrize(("pdf_number", "references"), REFERENCES_PER_PDF.items())
def test_every_citation_mark_in_the_sample_resolves(pdf_number: int, references: int) -> None:
    chunks = chunks_of(SAMPLES / f"Test Second Order References {pdf_number}.pdf")
    index = parse_reference_lists(rebuild_pages(chunks))
    marks = [(page, mark.numbers) for page, chunk in chunks for mark in find_marks(chunk)]

    unresolved = [
        (page, numbers) for page, numbers in marks if not index.resolve_mark(page, numbers)
    ]

    assert sorted(index.document) == list(range(1, references + 1)), "reference list incomplete"
    assert marks, "no citation marks found"
    assert unresolved == []
