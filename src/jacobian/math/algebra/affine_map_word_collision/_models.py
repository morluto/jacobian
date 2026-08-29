"""Typed contracts for the affine-map word collision profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_GENERATORS = 20
MAX_DEPTH = 10
MAX_WORDS = 10_000


class AffineMapSpec(StrictModel):
    """One affine map x -> slope*x + intercept."""

    slope: CanonicalRational
    intercept: CanonicalRational


class WordCollisionProfileRequest(StrictModel):
    """Request for the affine-map word collision profile."""

    generators: tuple[AffineMapSpec, ...] = Field(min_length=1)
    depth: int = Field(ge=1, le=MAX_DEPTH)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.generators) > MAX_GENERATORS:
            raise PydanticCustomError(
                "affine_map.too_many_generators",
                f"at most {MAX_GENERATORS} generators are supported",
            )
        word_count = len(self.generators) ** self.depth
        if word_count > MAX_WORDS:
            raise PydanticCustomError(
                "affine_map.word_count_exceeds_bound",
                f"r^d = {word_count} exceeds the {MAX_WORDS}-word limit",
            )
        return self


class CollisionRow(StrictModel):
    """One collision class: a distinct composed map and its source words."""

    slope: CanonicalRational
    intercept: CanonicalRational
    multiplicity: int
    words: tuple[tuple[int, ...], ...]


class WordCollisionProfileResult(StrictModel):
    """The complete word collision profile of an affine-map family."""

    generators: tuple[AffineMapSpec, ...]
    depth: int
    rows: tuple[CollisionRow, ...]


__all__ = [
    "MAX_DEPTH",
    "MAX_GENERATORS",
    "MAX_WORDS",
    "AffineMapSpec",
    "CollisionRow",
    "WordCollisionProfileRequest",
    "WordCollisionProfileResult",
]
