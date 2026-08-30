"""Typed contracts for the affine-map word collision profile operation."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_WORDS = 10_000
MAX_GENERATORS = MAX_WORDS
MAX_DEPTH = MAX_WORDS


class AffineMapSpec(StrictModel):
    """One affine map x -> slope*x + intercept."""

    slope: CanonicalRational
    intercept: CanonicalRational


class WordCollisionProfileRequest(StrictModel):
    """Request for the affine-map word collision profile."""

    generators: tuple[AffineMapSpec, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    depth: StrictInt = Field(ge=1, le=MAX_DEPTH)


class CollisionRow(StrictModel):
    """One collision class: a distinct composed map and its source words."""

    slope: CanonicalRational
    intercept: CanonicalRational
    multiplicity: StrictInt = Field(ge=1)
    words: tuple[tuple[StrictInt, ...], ...]


class WordCollisionProfileResult(StrictModel):
    """The complete word collision profile of an affine-map family."""

    generators: tuple[AffineMapSpec, ...]
    depth: StrictInt = Field(ge=1, le=MAX_DEPTH)
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
