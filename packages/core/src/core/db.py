"""Engine and session factory.

Deliberately model-free: `core` never imports `domain`, so the two packages stay siblings.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(settings.database_url, pool_size=10, max_overflow=5)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success, roll back on failure, always close."""
    with get_session_factory()() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_session() -> Iterator[Session]:
    """FastAPI dependency. Caller owns the commit."""
    with get_session_factory()() as session:
        yield session
