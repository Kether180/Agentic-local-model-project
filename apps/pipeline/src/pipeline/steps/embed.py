"""s3: chunks -> vectors. The expensive step, and the one worth scaling out."""

from core.embedding import get_embedder
from core.hashing import StepConfig
from domain import PipelineEvent, StepName

from pipeline.step import PipelineStep, StepResult

BATCH = 32


class EmbedStep(PipelineStep):
    step_name = StepName.EMBED
    step_version = 1

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        raw_chunks = cached[StepName.CHUNK]["chunks"]
        if not isinstance(raw_chunks, list):
            raise TypeError(f"chunk step produced {type(raw_chunks).__name__}, expected list")

        texts: list[str] = []
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                raise TypeError("chunk entries must be objects")
            text = chunk.get("text")
            texts.append(text if isinstance(text, str) else "")

        embedder = get_embedder()
        vectors: list[list[float]] = []
        # Batched so one slow round trip covers many chunks; Ollama has no batch API, this is
        # just a bigger `input` array.
        for start in range(0, len(texts), BATCH):
            vectors.extend(embedder.embed_batch(texts[start : start + BATCH]))

        result: StepResult = {
            "embeddings": [list(v) for v in vectors],
            "count": len(vectors),
        }
        return result

    def step_config(self) -> StepConfig:
        # Read off the embedder, so provider and dimensions are part of the cache key: swapping
        # model or provider must invalidate every cached vector, not silently mix widths.
        embedder = get_embedder()
        return {
            "model": embedder.model,
            "dimensions": embedder.dimensions,
            "batch": BATCH,
        }
