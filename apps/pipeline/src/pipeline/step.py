"""The step framework.

One worker per queue, chained entry -> exit. A step subclasses this, implements `compute()`, and
gets the rest for free. Two decisions live in the base class rather than in each step:

1. **Caching is here, not in every step.** Duplicating the same cache check into every step is how
   the checks drift apart. Subclasses implement `compute()` and never see the cache at all.
2. **Publish, then ack.** Acking the entry queue before publishing to the exit queue opens a window
   where a crash drops the document silently, with nothing left to redeliver. Publishing first can
   duplicate a message instead of losing one, and the step cache makes a duplicate cheap — the
   second pass is a cache hit at every step.
"""

import logging
from abc import ABC, abstractmethod

from core.db import session_scope
from core.hashing import StepConfig, cache_key, config_hash
from core.rabbit import connect, consume, declare, publish
from domain import PipelineEvent, StepName, StepStatus, next_step, queue_name
from domain.pg import PipelineStatus, StepCache
from pika.adapters.blocking_connection import BlockingChannel
from pydantic import JsonValue
from sqlalchemy import select

log = logging.getLogger(__name__)

StepResult = dict[str, JsonValue]
"""What a step returns: JSON-serializable, because it is cached in a JSONB column."""


class PipelineStep(ABC):
    step_name: StepName
    step_version: int = 1
    """Bump when `compute()` changes meaning — it is part of the cache key, so bumping it
    invalidates every cached result for this step without touching the table."""

    cacheable: bool = True
    """False for steps whose work is a side effect rather than a value.

    Replaying a cached result is only equivalent to re-running when `compute()` is pure. The sink
    writes rows for *this* document, so skipping it on a content-hash hit would leave a second
    upload of the same bytes with no chunks at all."""

    def __init__(self) -> None:
        # Set by run(); forwarding reuses the consumer's channel rather than dialling per message.
        self._channel: BlockingChannel | None = None

    @abstractmethod
    def compute(self, event: PipelineEvent, cached: dict[StepName, StepResult]) -> StepResult:
        """Do the work. `cached` holds the results of the upstream steps this one depends on."""

    def step_config(self) -> StepConfig:
        """Settings this step's output depends on. Changing one invalidates the cache."""
        return {}

    @property
    def entry_queue(self) -> str:
        return queue_name(self.step_name)

    @property
    def exit_queue(self) -> str | None:
        """The queue this step forwards to, or `None` if it is the sink and the chain ends."""
        nxt = next_step(self.step_name)
        return queue_name(nxt) if nxt else None

    def _load_upstream(self, event: PipelineEvent) -> dict[StepName, StepResult]:
        """Read prior steps' outputs back out of the cache table."""
        if not event.artifacts:
            return {}
        with session_scope() as session:
            rows = session.scalars(
                select(StepCache).where(StepCache.cache_key.in_(event.artifacts.values()))
            ).all()
            by_key = {row.cache_key: row.result for row in rows}
        return {step: by_key[key] for step, key in event.artifacts.items() if key in by_key}

    def _cached_result(self, key: str) -> StepResult | None:
        with session_scope() as session:
            row = session.get(StepCache, key)
            return dict(row.result) if row else None

    def _store_result(self, key: str, event: PipelineEvent, result: StepResult) -> None:
        with session_scope() as session:
            if session.get(StepCache, key) is None:
                session.add(
                    StepCache(
                        cache_key=key,
                        document_sha=event.sha256,
                        step_name=self.step_name,
                        step_version=self.step_version,
                        config_hash=config_hash(self.step_config()),
                        result=result,
                    )
                )

    def _record_status(
        self, event: PipelineEvent, status: StepStatus, cache_hit: bool, error: str = ""
    ) -> None:
        with session_scope() as session:
            session.add(
                PipelineStatus(
                    document_id=event.document_id,
                    step_name=self.step_name,
                    step_status=status,
                    cache_hit=cache_hit,
                    error_msg=error,
                )
            )

    def handle(self, body: str) -> None:
        event = PipelineEvent.model_validate_json(body)
        event.step_name = self.step_name
        event.step_status = StepStatus.IN_PROGRESS

        key = cache_key(
            event.sha256, self.step_name.value, self.step_version, config_hash(self.step_config())
        )

        cached = self._cached_result(key) if self.cacheable else None
        try:
            if cached is not None:
                log.info("cache hit  %s %s", self.step_name.value, event.document_id)
                result = cached
            else:
                log.info("cache miss %s %s — computing", self.step_name.value, event.document_id)
                result = self.compute(event, self._load_upstream(event))
                if self.cacheable:
                    self._store_result(key, event, result)
        except Exception as err:
            # Record the failure before re-raising. Without this the message dead-letters and the
            # document sits at `in_progress` forever with nothing saying why. Re-raising is what
            # nacks it to the DLQ — do not swallow it.
            log.exception("%s failed for %s", self.step_name.value, event.document_id)
            self._record_status(event, StepStatus.FAILED, cache_hit=False, error=str(err))
            raise

        event.artifacts[self.step_name] = key
        event.step_history.append(self.step_name)
        event.step_status = StepStatus.COMPLETED
        self._record_status(event, StepStatus.COMPLETED, cache_hit=cached is not None)

        self._forward(event)

    def _forward(self, event: PipelineEvent) -> None:
        """Publish to the exit queue.

        The entry-queue ack happens after this returns, in `core.rabbit.consume` —
        publish-then-ack, so a crash redelivers rather than drops.
        """
        if self.exit_queue is None:
            log.info("pipeline complete for %s", event.document_id)
            return
        if self._channel is None:  # pragma: no cover - only outside run(), e.g. in tests
            raise RuntimeError("no channel; call run() or set _channel")
        publish(self._channel, self.exit_queue, event.model_dump_json())

    def run(self) -> None:
        connection = connect()
        self._channel = connection.channel()
        queues = [self.entry_queue] + ([self.exit_queue] if self.exit_queue else [])
        declare(self._channel, *queues)
        consume(self._channel, self.entry_queue, self.handle)
