"""Reference extraction.

The model cites with `[S3]`; the client needs a real reference. This maps one to the other."""

import re

from api.crud.chunk import ChunkHit, CitedClaim, ReferenceSource
from api.schemas.responses import UrlCitation
from api.services.agent.references import sentence_before

MARKER = re.compile(r"\[S(\d+)\]")  # recognize a citation
WORD = re.compile(r"[a-z]{4,}")
STOPWORDS = frozenset(
    {"that", "this", "with", "from", "were", "have", "than", "been", "more", "which", "their"}
    | {"into", "also", "such", "these", "those", "there", "about", "other", "only", "most"}
)
MIN_SHARED_WORDS = 2


def _words(text: str) -> set[str]:
    words: list[str] = WORD.findall(text.lower())
    return {word.rstrip("s") for word in words if word not in STOPWORDS}


def attribute(sentence: str, claims: tuple[CitedClaim, ...]) -> list[ReferenceSource]:
    """Sources of the chunk sentence that the model's sentence most clearly restates.

    Word overlap, not meaning: a restated claim keeps its key terms ("waist", "height", "ratio",
    "adjunct"). Below `MIN_SHARED_WORDS` nothing is attributed and the caller falls back to the
    chunk itself, because a sentence from an uncited part of the chunk is the author's own claim,
    and a wrong source is worse than none.
    """
    said = _words(sentence)
    scores = [len(said & _words(claim.text)) for claim in claims]
    best = max(scores, default=0)
    if best < MIN_SHARED_WORDS:
        return []
    sources: dict[int, ReferenceSource] = {}
    for score, claim in zip(scores, claims, strict=True):
        if score == best:
            for source in claim.sources:
                sources.setdefault(source.number, source)
    return list(sources.values())


def parse_markers(
    text: str, hits: list[ChunkHit]
) -> list[UrlCitation]:  # turns the answer's labels into annotations.
    """Resolve every `[Sn]` in `text` to a citation, in the order they appear.

    Markers pointing past the end of `hits` are dropped: a small model will occasionally invent
    `[S9]` from a list of four, and a citation to a chunk that was never retrieved is worse than
    no citation. Repeats of the same marker are kept — each occurrence annotates its own span,
    which is what `start_index`/`end_index` are for.

    Second order: when the model's sentence restates a cited sentence of the chunk, the marker
    annotates that sentence's original sources instead (one citation each, same span), pointing at
    the page of the reference entry. Otherwise it keeps its first order citation to the chunk.
    """
    citations: list[UrlCitation] = []
    for match in MARKER.finditer(text):  # find every [s1] , [s2] in the answer
        index = int(match.group(1)) - 1  # [s4] position 3
        if not 0 <= index < len(hits):  # was there really a 4th search result?
            continue  # no.. ignore this label, no citation
        hit = hits[index]
        sources = attribute(MARKER.sub("", sentence_before(text, match.start())), hit.claims)
        if sources:
            cited_in = f"(cited in {hit.filename} p.{hit.page})"
            targets = [(s.page, f"[{s.number}] {s.text} {cited_in}") for s in sources]
        else:
            targets = [(hit.page, f"{hit.filename} p.{hit.page}")]
        citations.extend(
            UrlCitation(
                url=f"/documents/{hit.document_id}#page={page}",
                title=title,
                start_index=match.start(),
                end_index=match.end(),
            )
            for page, title in targets
        )
    return citations
