"""s1: PDF -> per-page text."""

import pymupdf
from core.blobs import get_blob_store
from core.hashing import StepConfig
from domain import PipelineEvent, StepName

from pipeline.step import PipelineStep, StepResult


class ParseStep(PipelineStep):
    step_name = StepName.PARSE
    step_version = 1

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        data = get_blob_store().get(event.blob_key)
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages: list[str] = [str(page.get_text()) for page in doc]
        result: StepResult = {"pages": list(pages), "page_count": len(pages)}
        return result

    def step_config(self) -> StepConfig:
        return {"extractor": "pymupdf-text"}
