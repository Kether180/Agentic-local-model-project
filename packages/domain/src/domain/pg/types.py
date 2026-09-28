"""Column types."""

from typing import Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import Dialect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

ModelT = TypeVar("ModelT", bound=BaseModel)


class PydanticJSONB(TypeDecorator[ModelT], Generic[ModelT]):
    """A JSONB column that reads back as its pydantic model rather than a raw mapping.

    Without this the attribute would be typed as some loose JSON shape and every reader would
    re-validate by hand.
    """

    impl = JSONB
    cache_ok = True

    def __init__(self, model: type[ModelT]) -> None:
        super().__init__()
        self.model = model

    def process_bind_param(self, value: ModelT | None, dialect: Dialect) -> object | None:
        return None if value is None else value.model_dump(mode="json")

    def process_result_value(self, value: object | None, dialect: Dialect) -> ModelT | None:
        return None if value is None else self.model.model_validate(value)

    def __repr__(self) -> str:
        """Alembic writes this into migrations, prefixed with this module's path.

        So the repr omits the prefix but fully qualifies the model; `script.py.mako` imports
        `domain.citation` and `domain.pg.types` to make both resolvable there.
        """
        model = self.model
        return f"PydanticJSONB({model.__module__}.{model.__qualname__})"
