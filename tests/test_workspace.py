"""Workspace wiring: every member imports and the packages stay independent."""

from pathlib import Path
from types import ModuleType
from uuid import UUID

import core
import domain
from core import Settings, settings
from domain import PipelineEvent, StepName, next_step, queue_name
from domain.pg import Base, Chunk
from pgvector.sqlalchemy import Vector

import pipeline


def test_settings_load():
    assert Settings().rabbit_url.startswith("amqp://")
    assert settings.chat.provider == "ollama"
    assert settings.embedding.provider == "ollama"


def test_core_and_domain_are_independent():
    """Neither package may import the other — apps wire them together."""
    for module in _module_files(core):
        assert "import domain" not in module, "core must not import domain"
    for module in _module_files(domain):
        assert "import core" not in module, "domain must not import core"


def _module_files(package: ModuleType) -> list[str]:
    source = package.__file__
    assert source is not None
    return [p.read_text() for p in Path(source).parent.rglob("*.py")]


def test_step_chain_is_linear_and_ends_at_index():
    assert queue_name(StepName.PARSE) == "q.parse"
    assert next_step(StepName.PARSE) is StepName.CHUNK
    assert next_step(StepName.INDEX) is None


def test_every_step_registered_and_wired():
    for name, step_cls in pipeline.STEPS.items():
        step = step_cls()
        assert step.entry_queue == queue_name(name)
    assert pipeline.STEPS[StepName.INDEX]().exit_queue is None


def test_embedding_dim_matches_the_column():
    """The configured embedder and the schema agree on vector width.

    `core` and `domain` cannot import each other, so this test is the seam that reconciles them.
    `EMBED_DIM` is defined once, in `domain`; `core` derives width per model. If they drift, every
    insert fails at the database — catch it here instead.
    """
    from domain.pg.models import EMBED_DIM as schema_dim

    column_type = Chunk.__table__.c.embedding.type
    assert isinstance(column_type, Vector)
    assert column_type.dim == schema_dim
    assert core.get_embedder().dimensions == schema_dim


def test_schema_has_the_four_tables():
    assert set(Base.metadata.tables) == {"document", "chunk", "step_cache", "pipeline_status"}


def test_pipeline_event_round_trips():
    event = PipelineEvent(
        document_id=UUID("0f9d1f6e-4d1a-4a1f-9c3e-3a2b1c4d5e6f"),
        sha256="a" * 64,
        filename="x.pdf",
        blob_key="a/x.pdf",
    )
    assert PipelineEvent.model_validate_json(event.model_dump_json()) == event
