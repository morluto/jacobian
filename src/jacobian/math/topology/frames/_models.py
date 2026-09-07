"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.frames.values import (
    MAX_DIM,
    MAX_VECTOR_CELLS,
    VectorFamily,
)


class VectorFamilyRequest(VectorFamily):
    """A bounded family in the standard ordered coordinate space."""


class FiniteFrameRequest(VectorFamilyRequest):
    """Wire request for an operation whose input must span its ambient space."""


class CoherenceRequest(FiniteFrameRequest):
    """Wire request for the normalized pairwise coherence operation."""


class GramResult(VectorFamilyRequest):
    gram: IntegerMatrix
    dimension: int = Field(ge=1)

    @model_validator(mode="after")
    def require_source_shape(self) -> Self:
        if (
            self.gram.row_count != len(self.vectors)
            or self.gram.column_count != len(self.vectors)
            or self.dimension != len(self.vectors[0])
        ):
            raise PydanticCustomError(
                "frames.gram_shape",
                "Gram matrix must align with the retained vector-family axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        gram: IntegerMatrix,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            gram=gram,
            dimension=len(vectors[0]),
        )


class CoherenceResult(CoherenceRequest):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        coherence_squared: CanonicalRational,
        maximizing_pair: tuple[int, int] | None,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            coherence_squared=coherence_squared,
            maximizing_pair=maximizing_pair,
        )


class FramePotentialResult(FiniteFrameRequest):
    potential: ExactInteger

    @classmethod
    def _from_kernel(
        cls, *, vectors: tuple[tuple[int, ...], ...], potential: ExactInteger
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            potential=potential,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceRequest",
    "CoherenceResult",
    "FiniteFrameRequest",
    "FramePotentialResult",
    "GramResult",
    "VectorFamilyRequest",
]
