"""Typed contracts for the affine-map word collision profile operation."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictInt

from jacobian._exact import CanonicalRational, format_canonical_rational
from jacobian._models import StrictModel

MAX_COMPOSITION_WORK = 5_000_000
MAX_GENERATORS = MAX_COMPOSITION_WORK
MAX_DEPTH = MAX_COMPOSITION_WORK


class AffineMapSpec(StrictModel):
    """One affine map x -> slope*x + intercept."""

    slope: CanonicalRational
    intercept: CanonicalRational

    @classmethod
    def _from_kernel(cls, slope: Fraction, intercept: Fraction) -> Self:
        def rational(value: Fraction) -> CanonicalRational:
            numerator, separator, denominator = format_canonical_rational(
                value
            ).partition("/")
            return CanonicalRational.model_construct(
                num=numerator, den=denominator if separator else "1"
            )

        return cls.model_construct(
            slope=rational(slope),
            intercept=rational(intercept),
        )


class WordCollisionProfileRequest(StrictModel):
    """Request for the affine-map word collision profile."""

    generators: tuple[AffineMapSpec, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    depth: StrictInt = Field(ge=1, le=MAX_DEPTH)


class CollisionRow(StrictModel):
    """One collision class: a distinct composed map and its source words."""

    map: AffineMapSpec
    multiplicity: StrictInt = Field(ge=1)
    words: tuple[tuple[StrictInt, ...], ...]

    @classmethod
    def _from_kernel(
        cls,
        slope: Fraction,
        intercept: Fraction,
        multiplicity: int,
        words: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            map=AffineMapSpec._from_kernel(slope, intercept),
            multiplicity=multiplicity,
            words=words,
        )


class WordCollisionProfileResult(StrictModel):
    """The complete word collision profile of an affine-map family."""

    generators: tuple[AffineMapSpec, ...]
    depth: StrictInt = Field(ge=1, le=MAX_DEPTH)
    rows: tuple[CollisionRow, ...]

    @classmethod
    def _from_kernel(
        cls,
        generators: tuple[AffineMapSpec, ...],
        depth: int,
        rows: tuple[CollisionRow, ...],
    ) -> Self:
        return cls.model_construct(generators=generators, depth=depth, rows=rows)


__all__ = [
    "MAX_COMPOSITION_WORK",
    "MAX_DEPTH",
    "MAX_GENERATORS",
    "AffineMapSpec",
    "CollisionRow",
    "WordCollisionProfileRequest",
    "WordCollisionProfileResult",
]
