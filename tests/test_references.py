"""Second order references: a chunk's citation marks must resolve to its references section.

Fixtures mirror the extracted text of the sample PDFs: superscripts arrive as plain digits, glued
to a word or after a full stop, and the false positives from the README sit in the same text.
"""

from uuid import uuid4

from api.crud.chunk import ChunkHit
from api.services.agent.citations import parse_markers
from api.services.agent.references import (
    attach_sources,
    expand,
    find_marks,
    parse_reference_lists,
    resolve,
)


def numbers(text: str) -> list[list[int]]:
    return [mark.numbers for mark in find_marks(text)]


SLIDE = """S EC T I O N 0 1  ·  D E F I NI T I O NS
Obesity is characterised by excess adipose accumulation
sufficient to impair health1-3. BMI performs inconsistently at
the extremes of stature2,4. Waist-to-height ratio is an adjunct measures5.
25.0 – 29.9*
BMI, kg/m²
*  The overweight band is not ancestry-adjusted in some populations5.
References
1. Halvorsen RT. Redefining adiposity thresholds. J Metab Med. 2021.
2. Okonkwo BA. Limitations of body mass index. Ann Obes Sci. 2020.
3. Ferreira-Lopes DN. Anthropometric screening. Int Rev Endocrinol. 2022.
4. Takahashi RE. Lean mass misclassification. J Popul Metab Health. 2021.
5. Ndiaye AB. Waist-to-height ratio as an adjunct
screening measure. Clin Pract Advis. 2022.
2"""


def test_ranges_and_lists_expand() -> None:
    assert expand("1-3") == [1, 2, 3]
    assert expand("2,4") == [2, 4]
    assert expand("31,34–37") == [31, 34, 35, 36, 37]


def test_superscript_marks_skip_the_readme_false_positives() -> None:
    """`29.9*`, `kg/m²` and the footnote's `populations5` are not marks; the list is not body."""
    assert numbers(SLIDE) == [[1, 2, 3], [2, 4], [5]]


def test_unit_exponent_extracted_as_a_plain_digit_is_not_a_mark() -> None:
    """PDFs 1 and 5 extract `kg/m²` as `kg/m2`; reference 2 exists, so only the unit rule stops it."""
    text = "adjustment.34,36,39 Category\nBMI (kg/m2)\nmean (SD), kg/m2\nstature2,4."

    assert numbers(text) == [[34, 36, 39], [2, 4]]


def test_footnote_wrapped_onto_a_second_line_is_still_skipped() -> None:
    text = "Adiposity rose sharply3.\n*  Lower cut-offs apply in some\npopulations5.\nBody again4."

    assert numbers(text) == [[3], [4]]


def test_footnote_ends_where_the_body_resumes() -> None:
    """No blank line in extracted text: `Abstract` starts with a capital, so the body is back."""
    text = "*Corresponding author: k.b@nordvest.example\nAbstract\nWeight fell in trials.12"

    assert numbers(text) == [[12]]


def test_bracketed_and_after_full_stop_marks() -> None:
    assert numbers("impair health [1]. At the extremes of stature [1,2].") == [[1], [1, 2]]
    assert numbers("misclassifies at the extremes of stature.31,34–37 Waist") == [
        [31, 34, 35, 36, 37]
    ]


def test_slide_list_resolves_with_wrapped_entry_and_no_slide_number() -> None:
    lists = parse_reference_lists({2: SLIDE})

    assert sorted(lists.pages[2]) == [1, 2, 3, 4, 5]
    assert lists.pages[2][5].text == (
        "Ndiaye AB. Waist-to-height ratio as an adjunct screening measure. Clin Pract Advis. 2022."
    )


def test_journal_list_continues_onto_a_page_without_heading() -> None:
    pages = {
        3: "Body text ends here.\nReferences\n[1] Sorensen VJ. Dietary composition.\n[2] Olesen PO.",
        4: "Journal of Metabolic Medicine\nHaugen et al.\n[3] Delgado EP. Weight stigma.\n",
    }

    lists = parse_reference_lists(pages)

    assert sorted(lists.document) == [1, 2, 3]
    assert lists.document[3].page == 4


def test_numbered_list_that_does_not_continue_is_not_a_reference_list() -> None:
    pages = {1: "References\n1. Halvorsen RT. Thresholds.", 2: "Steps\n7. Measure the waist."}

    assert sorted(parse_reference_lists(pages).document) == [1]


def test_letter_spaced_heading_and_consolidated_list() -> None:
    """PDF 3: marks on one slide resolve to a consolidated list on a later page."""
    pages = {
        6: "Costs are substantial [12,13].",
        11: "R EF E R E N C E S\n12. Rautenbach SJ. Direct costs.\n13. Kaur NS. Productivity.",
    }

    [claim] = resolve(pages[6], 6, parse_reference_lists(pages))

    assert [(s.number, s.page) for s in claim.sources] == [(12, 11), (13, 11)]


def test_overlapping_chunks_keep_the_longest_entry() -> None:
    page = "References\n2. Okonkwo BA. Limitations of body mass index.\n2. Okonkwo BA. Limit"

    assert parse_reference_lists({1: page}).pages[1][2].text == (
        "Okonkwo BA. Limitations of body mass index."
    )


def test_mark_that_resolves_nowhere_is_dropped() -> None:
    lists = parse_reference_lists({1: "References\n1. Halvorsen RT. Thresholds."})

    assert resolve("Seen in 1 of 3 cohorts, as shown by Study2.", 1, lists) == ()


def test_each_mark_keeps_the_sentence_it_is_attached_to() -> None:
    claims = resolve(SLIDE, 2, parse_reference_lists({2: SLIDE}))

    assert [[s.number for s in c.sources] for c in claims] == [[1, 2, 3], [2, 4], [5]]
    assert claims[1].text.startswith("BMI performs inconsistently")
    assert claims[2].text == "Waist-to-height ratio is an adjunct measures"


def slide_hit() -> ChunkHit:
    document_id = uuid4()
    hit = ChunkHit(
        chunk_id=uuid4(),
        document_id=document_id,
        filename="deck.pdf",
        page=2,
        text=SLIDE,
        distance=0.1,
    )
    [resolved] = attach_sources([hit], {document_id: {2: SLIDE}})
    return resolved


def test_annotation_cites_only_the_sources_of_the_restated_sentence() -> None:
    """The README page: only Ndiaye [5] supports a sentence about waist-to-height ratio."""
    hit = slide_hit()
    text = "Waist-to-height ratio is a useful adjunct measure [S1]."

    [citation] = parse_markers(text, [hit])

    assert citation.title.startswith("[5] Ndiaye AB")
    assert citation.url == f"/documents/{hit.document_id}#page=2"
    assert text[citation.start_index : citation.end_index] == "[S1]"


def test_each_marker_is_attributed_by_its_own_sentence() -> None:
    text = (
        "BMI misclassifies lean mass at the extremes of stature [S1]. "
        "Waist-to-height ratio is an adjunct measure [S1]."
    )

    citations = parse_markers(text, [slide_hit()])

    assert [c.title.split("]")[0] for c in citations] == ["[2", "[4", "[5"]


def test_sentence_matching_no_cited_claim_keeps_first_order_citation() -> None:
    """An uncited part of the chunk is the author's own claim: never pin it on another paper."""
    [citation] = parse_markers(
        "BMI is weight in kilograms over height squared [S1].", [slide_hit()]
    )

    assert citation.title == "deck.pdf p.2"


def test_chunk_without_marks_keeps_first_order_citation() -> None:
    hit = ChunkHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename="a.pdf",
        page=7,
        text="No marks.",
        distance=0.1,
    )

    [citation] = parse_markers("Claim [S1].", [hit])

    assert citation.title == "a.pdf p.7"
