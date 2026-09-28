"""`[S3]` in an answer must become a citation the client can follow.

This is the reference-extraction half of the agent, and it is the part most exposed to a small
model behaving badly — inventing a source number, or citing the same one twice. Both cases are
pinned here.
"""

from uuid import UUID, uuid4

from api.crud.chunk import ChunkHit
from api.services.agent.citations import parse_markers


def hit(filename: str, page: int, document_id: UUID | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=uuid4(),
        document_id=document_id or uuid4(),
        filename=filename,
        page=page,
        text="…",
        distance=0.1,
    )


def test_marker_resolves_to_its_chunk() -> None:
    hits = [hit("a.pdf", 1), hit("b.pdf", 7)]
    text = "Plaques form early [S2]."

    citations = parse_markers(text, hits)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.url == f"/documents/{hits[1].document_id}#page=7"
    assert citation.title == "b.pdf p.7"
    assert text[citation.start_index : citation.end_index] == "[S2]"


def test_every_occurrence_annotates_its_own_span() -> None:
    """Two mentions of one source are two annotations, because each marks a different span."""
    hits = [hit("a.pdf", 3)]
    text = "First [S1]. Second [S1]."

    citations = parse_markers(text, hits)

    assert len(citations) == 2
    assert [c.url for c in citations] == [c.url for c in citations[:1]] * 2
    assert citations[0].start_index < citations[1].start_index
    for citation in citations:
        assert text[citation.start_index : citation.end_index] == "[S1]"


def test_invented_marker_is_dropped() -> None:
    """A citation to a chunk that was never retrieved is worse than no citation at all."""
    hits = [hit("a.pdf", 1)]

    assert parse_markers("As shown [S9].", hits) == []
    assert parse_markers("As shown [S0].", hits) == []


def test_no_markers_and_no_hits() -> None:
    assert parse_markers("No sources here.", [hit("a.pdf", 1)]) == []
    assert parse_markers("Dangling [S1].", []) == []
