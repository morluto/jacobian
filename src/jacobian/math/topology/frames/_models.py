"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.frames.values import (
    MAX_DIM,
    MAX_VECTOR_CELLS,
    VectorFamily,
)


class GramResult(VectorFamily):
    gram: IntegerMatrix

    @model_validator(mode="after")
    def require_source_shape(self) -> Self:
        if self.gram.row_count != len(self.vectors) or self.gram.column_count != len(
            self.vectors
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
        dimension: int,
        gram: IntegerMatrix,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            gram=gram,
            dimension=dimension,
        )


class CoherenceResult(VectorFamily):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        coherence_squared: CanonicalRational,
        maximizing_pair: tuple[int, int] | None,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            dimension=dimension,
            coherence_squared=coherence_squared,
            maximizing_pair=maximizing_pair,
        )


class FramePotentialResult(VectorFamily):
    potential: ExactInteger

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        dimension: int,
        potential: ExactInteger,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            dimension=dimension,
            potential=potential,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceResult",
    "FramePotentialResult",
    "GramResult",
]
