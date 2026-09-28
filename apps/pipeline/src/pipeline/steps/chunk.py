"""s2: page text -> chunks sized for the embedding model."""

from core.embedding import get_embedder
from core.hashing import StepConfig
from domain import PipelineEvent, StepName
from pydantic import JsonValue

from pipeline.step import PipelineStep, StepResult

CHUNK_CHARS = 1200
"""~4 chars per token, so a chunk stays well inside the embedding model's context even in the
worst case. Tokenizing to measure exactly would mean shipping the tokenizer for one number."""

OVERLAP_CHARS = 150


def split(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Fixed-width windows with overlap, preferring a paragraph or sentence break near the edge."""
    if size <= overlap:
        raise ValueError(f"size ({size}) must exceed overlap ({overlap})")
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        window = text[start:end]
        if end < length:
            # Back up to the last clean break so chunks do not end mid-sentence. `end` moves with
            # the window, otherwise the next start is computed from a boundary we did not use.
            for marker in ("\n\n", "\n", ". "):
                cut = window.rfind(marker)
                if cut > size // 2:
                    window = window[: cut + len(marker)]
                    end = start + len(window)
                    break
        stripped = window.strip()
        if stripped:
            chunks.append(stripped)
        if end >= length:
            break
        # size > overlap is enforced above, so this always advances.
        start = end - overlap
    return chunks


class ChunkStep(PipelineStep):
    step_name = StepName.CHUNK
    step_version = 1

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        parsed = cached[StepName.PARSE]
        pages = parsed["pages"]
        if not isinstance(pages, list):
            raise TypeError(f"parse step produced {type(pages).__name__}, expected list")

        chunks: list[JsonValue] = []
        for page_number, page_text in enumerate(pages, start=1):
            if not isinstance(page_text, str):
                continue
            for piece in split(page_text):
                chunks.append({"ordinal": len(chunks), "page": page_number, "text": piece})
        result: StepResult = {"chunks": chunks, "chunk_count": len(chunks)}
        return result

    def step_config(self) -> StepConfig:
        return {
            "chunk_chars": CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "embed_context": get_embedder().context_tokens,
        }
