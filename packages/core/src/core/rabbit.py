"""RabbitMQ helpers shared by the api (publishes) and the pipeline (consumes).

Topology is the simplest thing that carries a linear chain: one durable work queue per step on the
default exchange, routing key == queue name. No topic exchange, because nothing fans out — each
step has exactly one successor.

Failure policy is deliberately unforgiving: a handler that raises gets `basic_nack(requeue=False)`,
which the queue's dead-letter config routes straight to the shared DLQ. Retrying a poisoned message
in place would just spin, and the step cache already makes a deliberate redelivery cheap.
"""

import logging
import time
from collections.abc import Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError
from pika.spec import Basic, BasicProperties

from core.config import settings

log = logging.getLogger(__name__)

DLQ = "q.dlq"

_QUEUE_ARGS = {"x-dead-letter-exchange": "", "x-dead-letter-routing-key": DLQ}


def connect(url: str | None = None, attempts: int = 30, delay: float = 2.0):
    """Blocking connect with retry — compose starts workers before the broker finishes booting."""
    params = pika.URLParameters(url or settings.rabbit_url)
    last: AMQPConnectionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return pika.BlockingConnection(params)
        except AMQPConnectionError as err:  # pragma: no cover - timing dependent
            last = err
            log.warning("rabbit not ready (%s/%s), retrying in %ss", attempt, attempts, delay)
            time.sleep(delay)
    raise RuntimeError(f"could not reach rabbit after {attempts} attempts") from last


def declare(channel: BlockingChannel, *queues: str) -> None:
    """Declare work queues plus the shared DLQ. Idempotent, so every worker can call it."""
    channel.queue_declare(queue=DLQ, durable=True)
    for queue in queues:
        channel.queue_declare(queue=queue, durable=True, arguments=_QUEUE_ARGS)


def publish(channel: BlockingChannel, queue: str, body: str) -> None:
    """Publish an already-serialized JSON body.

    Taking a string rather than a model keeps `core` free of any `domain` import — the caller
    owns (de)serialization, which is one `model_dump_json()` at the call site.
    """
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=body.encode(),
        properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
    )


def consume(channel: BlockingChannel, queue: str, handler: Callable[[str], None]) -> None:
    """Consume forever. `prefetch_count=1` so a scaled-out step spreads work evenly."""

    def on_message(
        ch: BlockingChannel,
        method: Basic.Deliver,
        _properties: BasicProperties,
        body: bytes,
    ) -> None:
        try:
            handler(body.decode())
        except Exception:
            log.exception("handler failed, dead-lettering message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        else:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue, on_message_callback=on_message)
    log.info("consuming %s", queue)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:  # pragma: no cover - signal path
        channel.stop_consuming()
