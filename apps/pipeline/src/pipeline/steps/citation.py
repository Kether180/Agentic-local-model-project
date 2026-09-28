"""s4: document-level citation metadata, via an LLM."""

import logging

from core.chat import get_chat_model
from core.config import settings
from core.hashing import StepConfig
from domain import CitationMetadata, PipelineEvent, StepName

from pipeline.step import PipelineStep, StepResult

log = logging.getLogger(__name__)

SIZE = "small"
HEAD_CHARS = 4000
"""Citation data lives on the first page or two; sending the whole document wastes tokens."""

PROMPT = (
    "Extract bibliographic citation metadata from the start of this document. "
    "Use null for anything not clearly stated — do not guess.\n\n{head}"
)


class CitationStep(PipelineStep):
    step_name = StepName.CITATION
    step_version = 1

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        pages = cached[StepName.PARSE]["pages"]
        if not isinstance(pages, list):
            raise TypeError(f"parse step produced {type(pages).__name__}, expected list")
        head = "\n".join(p for p in pages if isinstance(p, str))[:HEAD_CHARS]

        model = get_chat_model(SIZE).with_structured_output(CitationMetadata)
        try:
            response = model.invoke(PROMPT.format(head=head))
            citation = CitationMetadata.model_validate(response)
        except Exception:
            # A small local model failing to emit valid JSON must not sink the document — the
            # chunks and vectors are still worth indexing.
            log.exception("citation extraction failed for %s, continuing empty", event.document_id)
            citation = CitationMetadata()

        return {"citation": citation.model_dump(mode="json")}

    def step_config(self) -> StepConfig:
        cfg = settings.chat
        config: dict[str, str | int | float | bool] = {
            "provider": cfg.provider,
            "model": cfg.chat_models[SIZE],
            "head_chars": HEAD_CHARS,
        }
        if cfg.provider == "ollama":
            config["reasoning"] = cfg.reasoning
        return config
