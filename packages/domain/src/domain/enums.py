from enum import StrEnum


class StepName(StrEnum):
    """The pipeline stages, in order. The value is also the queue suffix (`q.parse`, ...)."""

    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    CITATION = "citation"
    INDEX = "index"


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# The chain. `None` as the exit queue marks the sink.
STEP_ORDER: list[StepName] = [
    StepName.PARSE,
    StepName.CHUNK,
    StepName.EMBED,
    StepName.CITATION,
    StepName.INDEX,
]


def queue_name(step: StepName) -> str:
    return f"q.{step.value}"


def next_step(step: StepName) -> StepName | None:
    index = STEP_ORDER.index(step)
    return STEP_ORDER[index + 1] if index + 1 < len(STEP_ORDER) else None
