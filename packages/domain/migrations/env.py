"""Alembic environment.

The database URL is a deployment detail, so it is read from the environment here rather than
committed to config. `domain` stays independent of `core`, which is why this reads the env var
directly instead of importing `core.config`.
"""

import os
from logging.config import fileConfig

from alembic import context
from domain.pg import Base
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

DEFAULT_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/take_home"


def database_url() -> str:
    """`APP_DATABASE_URL` is what the app and compose set; `DATABASE_URL` is the plain override."""
    return os.getenv("DATABASE_URL") or os.getenv("APP_DATABASE_URL") or DEFAULT_URL


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
