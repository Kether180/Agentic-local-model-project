"""A step must compute on a miss and replay on a hit.

This is the behaviour the whole design exists for: a re-import of unchanged bytes must not call
the embedding model again. Backed by a fake session so the test needs no database.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from core.hashing import StepConfig
from domain import PipelineEvent, StepName, StepStatus
from domain.pg import PipelineStatus, StepCache

from pipeline.step import PipelineStep, StepResult


class FakeSession:
    """Just enough Session for the cache path: get by primary key, add, and scan."""

    def __init__(self, cache: dict[str, StepCache], statuses: list[PipelineStatus]) -> None:
        self.cache = cache
        self.statuses = statuses

    def get(self, model: type, key: str) -> StepCache | None:
        return self.cache.get(key) if model is StepCache else None

    def add(self, obj: object) -> None:
        if isinstance(obj, StepCache):
            self.cache[obj.cache_key] = obj
        elif isinstance(obj, PipelineStatus):
            self.statuses.append(obj)

    def scalars(self, _stmt: object) -> "FakeResult":
        # _load_upstream filters by key afterwards, so handing back everything is correct.
        return FakeResult(list(self.cache.values()))


class FakeResult:
    def __init__(self, rows: list[StepCache]) -> None:
        self._rows = rows

    def all(self) -> list[StepCache]:
        return self._rows


class CountingStep(PipelineStep):
    step_name = StepName.EMBED
    step_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.forwarded: list[str] = []

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        self.calls += 1
        return {"embeddings": [[0.1, 0.2]], "count": 1}

    def step_config(self) -> StepConfig:
        return {"model": "nomic-embed-text:v1.5"}

    def _forward(self, event: PipelineEvent) -> None:
        self.forwarded.append(event.model_dump_json())


@dataclass
class Store:
    """What the fake session persisted, so tests can assert on it."""

    cache: dict[str, StepCache] = field(default_factory=dict)
    statuses: list[PipelineStatus] = field(default_factory=list)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Store:
    persisted = Store()

    @contextmanager
    def fake_scope(url: str | None = None) -> Iterator[FakeSession]:
        yield FakeSession(persisted.cache, persisted.statuses)

    monkeypatch.setattr("pipeline.step.session_scope", fake_scope)
    return persisted


def event_for(sha: str) -> str:
    return PipelineEvent(
        document_id=uuid4(), sha256=sha, filename="doc.pdf", blob_key=f"{sha}/doc.pdf"
    ).model_dump_json()


def test_miss_computes_then_hit_replays(store: Store) -> None:
    step = CountingStep()
    body = event_for("a" * 64)

    step.handle(body)
    assert step.calls == 1, "first run is a miss and must compute"

    step.handle(body)
    assert step.calls == 1, "identical bytes must replay from cache, not recompute"

    assert len(store.cache) == 1
    assert [s.cache_hit for s in store.statuses] == [False, True]


def test_different_content_recomputes(store: Store) -> None:
    step = CountingStep()
    step.handle(event_for("a" * 64))
    step.handle(event_for("b" * 64))
    assert step.calls == 2, "different bytes must not share a cache entry"


def test_bumped_step_version_recomputes(store: Store) -> None:
    step = CountingStep()
    body = event_for("a" * 64)
    step.handle(body)

    step.step_version = 2
    step.handle(body)
    assert step.calls == 2, "a version bump must invalidate the cached result"


def test_forwards_with_its_artifact_recorded(store: Store) -> None:
    step = CountingStep()
    step.handle(event_for("a" * 64))

    forwarded = PipelineEvent.model_validate_json(step.forwarded[0])
    assert StepName.EMBED in forwarded.artifacts
    assert forwarded.step_history == [StepName.EMBED]


def test_upstream_results_are_loaded_from_cache(store: Store) -> None:
    """A step reads its predecessors' outputs out of the same cache table."""
    store.cache["upstream-key"] = StepCache(
        cache_key="upstream-key",
        document_sha="a" * 64,
        step_name=StepName.CHUNK,
        step_version=1,
        config_hash="x",
        result={"chunks": [{"ordinal": 0, "page": 1, "text": "hello"}]},
    )

    seen: dict[StepName, StepResult] = {}

    class Recording(CountingStep):
        def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
            seen.update(cached)
            return super().compute(event, cached)

    event = PipelineEvent(
        document_id=uuid4(),
        sha256="a" * 64,
        filename="doc.pdf",
        blob_key="k",
        artifacts={StepName.CHUNK: "upstream-key"},
    )
    Recording().handle(event.model_dump_json())

    chunks = seen[StepName.CHUNK]["chunks"]
    assert isinstance(chunks, list)
    first = chunks[0]
    assert isinstance(first, dict)
    assert first["text"] == "hello"


class SinkStep(CountingStep):
    """A step whose work is a side effect, so it must run every time."""

    cacheable = False


def test_uncacheable_step_always_computes(store: Store) -> None:
    step = SinkStep()
    body = event_for("a" * 64)

    step.handle(body)
    step.handle(body)

    assert step.calls == 2, "a side-effecting step must not be skipped on a content-hash hit"
    assert store.cache == {}, "and must not populate the cache"


class ExplodingStep(CountingStep):
    """A step whose work raises, to check the failure is recorded before it dead-letters."""

    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        self.calls += 1
        raise RuntimeError("ollama unreachable")


def test_failure_is_recorded_and_reraised(store: Store) -> None:
    step = ExplodingStep()

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        step.handle(event_for("a" * 64))

    assert [s.step_status for s in store.statuses] == [StepStatus.FAILED]
    assert "ollama unreachable" in store.statuses[0].error_msg
    assert store.cache == {}, "a failed step must not cache anything"
    assert step.forwarded == [], "a failed step must not forward"
