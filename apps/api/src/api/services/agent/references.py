"""Second order references: citation marks inside a chunk -> entries of the references section.

Deterministic and model-free: the retrieved documents' text is parsed on each request. A mark only
resolves if every number in it exists in the document's reference list. That one rule filters most
false positives (decimals, units, slide numbers) without special cases.
"""

import re
from dataclasses import dataclass, field, replace
from uuid import UUID

from api.crud.chunk import ChunkHit, CitedClaim, ReferenceSource

ENTRY_STYLES = (
    re.compile(r"^\s*\[(\d{1,3})\]\s+(.*)$"),  # journal: "[12] Author..."
    re.compile(r"^\s*(\d{1,3})\.\s+(\S.*)$"),  # slide: "12. Author..."
)
FOOTNOTE = re.compile(r"^\s*[*†‡]")
BRACKETED = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
# Superscripts extract as plain digits, glued to a word ("health1-3") or after its full stop
# ("stature.31,34–37"). A digit before the full stop is a decimal ("29.9*"), so it never matches.
# A digit on a unit after a slash is an exponent that lost its superscript ("kg/m2"), not a mark.
SUPERSCRIPT = re.compile(
    r"(?:(?<=[A-Za-z%)])|(?<=[A-Za-z%)]\.))"
    r"(?<!/[A-Za-z])(?<!/[A-Za-z]{2})(?<!/[A-Za-z]{3})"
    r"(\d{1,3}(?:[,\-–]\d{1,3})*)(?![\d*A-Za-z])"
)
MAX_RANGE = 50


@dataclass
class ReferenceIndex:
    """A document's reference entries: per page, and merged for the whole document."""

    pages: dict[int, dict[int, ReferenceSource]] = field(default_factory=dict)
    document: dict[int, ReferenceSource] = field(default_factory=dict)

    def add(self, page: int, entries: dict[int, ReferenceSource]) -> None:
        self.pages[page] = entries
        for number, entry in entries.items():
            known = self.document.get(number)
            if known is None or len(entry.text) > len(known.text):
                self.document[number] = entry

    def resolve_mark(self, page: int, numbers: list[int]) -> list[ReferenceSource]:
        """Entries for every number of one mark, or none if any number is not a listed reference.

        The list on the same page wins (slides number per slide); otherwise the document's.
        """
        found: list[ReferenceSource] = []
        for number in numbers:
            source = self.pages.get(page, {}).get(number) or self.document.get(number)
            if source is None:
                return []
            found.append(source)
        return found


def is_heading(line: str) -> bool:
    """`References`, also letter-spaced as some PDFs extract it (`R EF E R E N C E S`)."""
    return "".join(line.split()).lower() == "references"


def _entry_style(lines: list[str]) -> re.Pattern[str] | None:
    for line in lines:
        for style in ENTRY_STYLES:
            if style.match(line):
                return style
    return None


def _first_number(lines: list[str], style: re.Pattern[str]) -> int | None:
    for line in lines:
        match = style.match(line)
        if match:
            return int(match.group(1))
    return None


def _collect(lines: list[str], style: re.Pattern[str], page: int) -> dict[int, ReferenceSource]:
    copies: list[tuple[int, list[str]]] = []
    for line in lines:
        match = style.match(line)
        if match:
            copies.append((int(match.group(1)), [match.group(2).strip()]))
        elif copies and line.strip() and not line.strip().isdigit():  # skip slide numbers
            copies[-1][1].append(line.strip())
    entries: dict[int, str] = {}
    for number, parts in copies:
        text = re.sub(r"-\s+", "-", " ".join(parts))
        # Overlapping chunks repeat entries, one copy cut short at a chunk edge: keep the longest.
        if len(text) > len(entries.get(number, "")):
            entries[number] = text
    return {n: ReferenceSource(number=n, text=t, page=page) for n, t in entries.items()}


def parse_reference_lists(pages: dict[int, str]) -> ReferenceIndex:
    """Reference entries from a `References` heading, or from a list continuing onto a page."""
    index = ReferenceIndex()
    style: re.Pattern[str] | None = None
    last_number = 0
    for page in sorted(pages):
        lines = pages[page].splitlines()
        headings = [i for i, line in enumerate(lines) if is_heading(line)]
        if headings:
            lines = lines[headings[0] + 1 :]
            style = _entry_style(lines)
        elif style is not None and _first_number(lines, style) != last_number + 1:
            # Without a heading, a list only counts if its numbering carries on from the last page.
            style = None
        if style is None:
            continue
        entries = _collect(lines, style, page)
        if entries:
            index.add(page, entries)
            last_number = max(entries)
    return index


def expand(mark: str) -> list[int]:
    """`1-3` -> [1, 2, 3]; `2,4` -> [2, 4]; `31,34–37` -> [31, 34, 35, 36, 37]."""
    numbers: list[int] = []
    for part in mark.split(","):
        bounds = [int(b) for b in re.split(r"\s*[-–]\s*", part.strip()) if b]
        if len(bounds) == 2 and 0 < bounds[1] - bounds[0] <= MAX_RANGE:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            numbers.extend(bounds)
    return numbers


def _body(text: str) -> str:
    """The chunk's body on one line, without the references list, footnotes (a mark there is
    third order) and reference entries. Extracted text has no blank line after a footnote, so a
    footnote only wraps while its sentence is unfinished and the next line starts lowercase."""
    kept: list[str] = []
    footnote_open = False
    for line in text.splitlines():
        if is_heading(line):
            break
        stripped = line.strip()
        if FOOTNOTE.match(line) or (footnote_open and stripped[:1].islower()):
            footnote_open = not stripped.endswith((".", "!", "?"))
            continue
        footnote_open = False
        if not _entry_style([line]):
            kept.append(stripped)
    return " ".join(kept)


def sentence_before(text: str, end: int, floor: int = 0) -> str:
    """The sentence ending at `end`: back to the previous sentence break, or to `floor`."""
    before = text[floor:end]
    breaks = [before.rfind(mark) for mark in (". ", "? ", "! ", "\n")]
    return before[max(breaks) + 1 :].strip()


@dataclass(frozen=True)
class Mark:
    numbers: list[int]
    claim: str  # the sentence the mark is attached to


def find_marks(text: str) -> list[Mark]:
    """Citation marks in a chunk's body, each expanded to its reference numbers."""
    body = _body(text)
    matches = sorted(
        (match for pattern in (BRACKETED, SUPERSCRIPT) for match in pattern.finditer(body)),
        key=lambda match: match.start(),
    )
    marks: list[Mark] = []
    previous_end = 0
    for match in matches:
        claim = sentence_before(body, match.start(), floor=previous_end)
        marks.append(Mark(numbers=expand(match.group(1)), claim=claim))
        previous_end = match.end()
    return marks


def resolve(text: str, page: int, index: ReferenceIndex) -> tuple[CitedClaim, ...]:
    """The chunk's cited sentences, each with the entries its mark resolves to."""
    claims: list[CitedClaim] = []
    for mark in find_marks(text):
        sources = index.resolve_mark(page, mark.numbers)
        if sources:
            claims.append(CitedClaim(text=mark.claim, sources=tuple(sources)))
    return tuple(claims)


def attach_sources(hits: list[ChunkHit], pages: dict[UUID, dict[int, str]]) -> list[ChunkHit]:
    """Copy of `hits` with each hit's cited sentences resolved."""
    indexes = {doc: parse_reference_lists(doc_pages) for doc, doc_pages in pages.items()}
    resolved: list[ChunkHit] = []
    for hit in hits:
        index = indexes.get(hit.document_id)
        if index is not None:
            hit = replace(hit, claims=resolve(hit.text, hit.page, index))
        resolved.append(hit)
    return resolved
