"""Typed contracts for the affine-map word collision profile operation."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
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
            return CanonicalRational.model_construct(
                num=value.numerator, den=value.denominator
            )

        return cls.model_construct(
            slope=rational(slope),
            intercept=rational(intercept),
        )


class AffineMapFamily(StrictModel):
    """A finite ordered family of affine maps over one rational field."""

    generators: tuple[AffineMapSpec, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )


class WordCollisionProfileRequest(StrictModel):
    """Request for the affine-map word collision profile."""

    family: AffineMapFamily
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

    family: AffineMapFamily
    depth: StrictInt = Field(ge=1, le=MAX_DEPTH)
    rows: tuple[CollisionRow, ...]

    @model_validator(mode="after")
    def require_word_shapes(self) -> Self:
        generator_count = len(self.family.generators)
        for row in self.rows:
            if any(
                len(word) != self.depth
                or any(index < 0 or index >= generator_count for index in word)
                for word in row.words
            ):
                raise ValueError("collision words must be depth-sized family indices")
        return self

    @classmethod
    def _from_kernel(
        cls,
        family: AffineMapFamily,
        depth: int,
        rows: tuple[CollisionRow, ...],
    ) -> Self:
        return cls.model_construct(family=family, depth=depth, rows=rows)


__all__ = [
    "MAX_COMPOSITION_WORK",
    "MAX_DEPTH",
    "MAX_GENERATORS",
    "AffineMapFamily",
    "AffineMapSpec",
    "CollisionRow",
    "WordCollisionProfileRequest",
    "WordCollisionProfileResult",
]
