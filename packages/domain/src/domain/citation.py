"""Citation value objects.

Small pieces composed into a wide aggregate — `DateStruct` is reused for three different dates and
`Author` for both authors and editors, which is the whole reason they are their own types.
"""

from collections.abc import Mapping
from typing import Self, TypeAlias, Union

from pydantic import BaseModel, model_validator


class DateStruct(BaseModel):
    """A partially-known date. Accepts an ISO string and explodes it."""

    year: int | None = None
    month: int | None = None
    day: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_iso_string(cls, value: "DateInput") -> "DateInput":
        """Accepts what an LLM actually emits for a date: an ISO string or a mapping."""
        if not isinstance(value, str):
            return value
        parts = value.strip().split("-")[:3]
        keys = ("year", "month", "day")
        return {k: int(p) for k, p in zip(keys, parts, strict=False) if p.isdigit()}

    def __str__(self) -> str:
        return "-".join(f"{p:02d}" for p in (self.year, self.month, self.day) if p is not None)


DateInput: TypeAlias = Union[str, Mapping[str, int | None], "DateStruct", None]
"""What `DateStruct` will accept before validation."""


class Author(BaseModel):
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None

    @property
    def full_name(self) -> str:
        return " ".join(p for p in (self.first_name, self.middle_name, self.last_name) if p)


class CitationMetadata(BaseModel):
    """Document-level metadata extracted by the citation step."""

    title: str | None = None
    authors: list[Author] = []
    editors: list[Author] = []
    publication_date: DateStruct | None = None
    access_date: DateStruct | None = None
    journal: str | None = None
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    language: str | None = None

    @model_validator(mode="after")
    def _strip_blanks(self) -> Self:
        for field in ("title", "journal", "publisher", "doi", "url"):
            value: str | None = getattr(self, field)
            if value is not None and not value.strip():
                setattr(self, field, None)
        return self
