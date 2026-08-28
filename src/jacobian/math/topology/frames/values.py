"""Canonical finite vector-family values used by frame operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_VECTORS, MAX_DIM, MAX_VALUE = 32, 16, 1000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"frames.{reason}", message)


class VectorFamily(StrictModel):
    """A bounded family in the standard ordered coordinate space.

    Spanning and nonzero-vector requirements are operation-specific admission
    decisions made by the native kernels.
    """

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


__all__ = ["MAX_DIM", "MAX_VALUE", "MAX_VECTORS", "VectorFamily"]
