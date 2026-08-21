"""Provider-independent values for exact formal concept analysis."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_OBJECTS = 64
MAX_ATTRIBUTES = 64


class FormalContext(StrictModel):
    """An immutable finite formal context K = (G, M, I).

    ``objects`` is a tuple of unique object labels.  ``attributes`` is a tuple
    of unique attribute labels.  ``incidence`` is a tuple of ``(object_index,
    attribute_index)`` pairs, each denoting that object ``objects[oi]`` has
    attribute ``attributes[ai]``.
    """

    objects: tuple[str, ...] = Field(min_length=1, max_length=MAX_OBJECTS)
    attributes: tuple[str, ...] = Field(min_length=1, max_length=MAX_ATTRIBUTES)
    incidence: tuple[tuple[int, int], ...] = Field(default=())

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(set(self.objects)) != len(self.objects):
            raise ValueError("object labels must be unique")
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("attribute labels must be unique")
        seen: set[tuple[int, int]] = set()
        for oi, ai in self.incidence:
            if not 0 <= oi < len(self.objects):
                raise ValueError("incidence object index out of range")
            if not 0 <= ai < len(self.attributes):
                raise ValueError("incidence attribute index out of range")
            pair = (oi, ai)
            if pair in seen:
                raise ValueError("incidence pairs must be duplicate-free")
            seen.add(pair)
        return self


__all__ = [
    "MAX_ATTRIBUTES",
    "MAX_OBJECTS",
    "FormalContext",
]
