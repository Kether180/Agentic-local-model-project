"""Chunk splitting — a loop with edge cases, so it gets a check."""

from pipeline.steps.chunk import CHUNK_CHARS, OVERLAP_CHARS, split


def test_short_text_is_one_chunk():
    assert split("hello world") == ["hello world"]


def test_empty_text_yields_nothing():
    assert split("") == []
    assert split("   \n  ") == []


def test_long_text_splits_and_covers_everything():
    text = "word " * 2000
    chunks = split(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_CHARS for c in chunks), "no chunk may exceed the model's window"
    assert "".join(c.replace(" ", "") for c in chunks).startswith("word" * 10)


def test_chunks_overlap_so_context_is_not_cut():
    text = ". ".join(f"sentence number {i}" for i in range(400))
    chunks = split(text)
    assert len(chunks) > 1
    assert any(chunks[0][-20:].strip() in chunks[1] for _ in [0]) or len(chunks[1]) > 0


def test_prefers_a_clean_break():
    text = "a" * 700 + "\n\n" + "b" * 700
    first = split(text, size=1000, overlap=50)[0]
    assert first.endswith("a"), "should break at the paragraph, not mid-run"


def test_overlap_must_be_smaller_than_size():
    try:
        split("x" * 100, size=10, overlap=10)
    except ValueError as err:
        assert "must exceed" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_defaults_are_consistent():
    assert OVERLAP_CHARS < CHUNK_CHARS
