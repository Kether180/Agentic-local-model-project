from domain import StepName

from pipeline.step import PipelineStep
from pipeline.steps.chunk import ChunkStep
from pipeline.steps.citation import CitationStep
from pipeline.steps.embed import EmbedStep
from pipeline.steps.index import IndexStep
from pipeline.steps.parse import ParseStep

STEPS: dict[StepName, type[PipelineStep]] = {
    StepName.PARSE: ParseStep,
    StepName.CHUNK: ChunkStep,
    StepName.EMBED: EmbedStep,
    StepName.CITATION: CitationStep,
    StepName.INDEX: IndexStep,
}

__all__ = [
    "STEPS",
    "ChunkStep",
    "CitationStep",
    "EmbedStep",
    "IndexStep",
    "ParseStep",
]
