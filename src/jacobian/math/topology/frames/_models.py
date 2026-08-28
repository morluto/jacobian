"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel

MAX_VECTORS, MAX_DIM, MAX_VALUE = 32, 16, 1000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by frame contracts."""

    return PydanticCustomError(f"frames.{reason}", message)


class VectorFamilyRequest(StrictModel):
    """A bounded family in the standard ordered coordinate space."""

    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_VECTORS)

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        dimension = len(self.vectors[0])
        if not 1 <= dimension <= MAX_DIM:
            raise _validation_error(
                "vector_dimension_out_of_range",
                f"vector dimension must be between 1 and {MAX_DIM}",
            )
        if any(len(vector) != dimension for vector in self.vectors):
            raise _validation_error(
                "vector_dimension_mismatch", "all vectors must have equal dimension"
            )
        if any(abs(entry) > MAX_VALUE for vector in self.vectors for entry in vector):
            raise _validation_error(
                "vector_entry_out_of_range", "vector entries must be bounded"
            )
        return self


class FiniteFrameRequest(VectorFamilyRequest):
    """A vector family spanning its full standard ambient space."""


class CoherenceRequest(FiniteFrameRequest):
    """A finite frame whose normalized pairwise coherence is requested."""


class GramResult(VectorFamilyRequest):
    gram: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    method: str = "DOT_PRODUCT"

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        gram: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            gram=gram,
            dimension=len(vectors[0]),
            method="DOT_PRODUCT",
        )


class CoherenceResult(CoherenceRequest):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None
    method: str = "EXACT_MAX_SQUARED_NORMALIZED_INNER_PRODUCT"

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
            method="EXACT_MAX_SQUARED_NORMALIZED_INNER_PRODUCT",
        )


class FramePotentialResult(FiniteFrameRequest):
    potential: CanonicalInteger
    method: str = "EXACT_GRAM_SQUARE_SUM"

    @classmethod
    def _from_kernel(
        cls, *, vectors: tuple[tuple[int, ...], ...], potential: CanonicalInteger
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            potential=potential,
            method="EXACT_GRAM_SQUARE_SUM",
        )


__all__ = [
    "CoherenceRequest",
    "CoherenceResult",
    "FiniteFrameRequest",
    "FramePotentialResult",
    "GramResult",
    "VectorFamilyRequest",
]
