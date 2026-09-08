"""Canonical finite vector-family values used by frame operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_VECTOR_CELLS = 524_288
MAX_DIM = MAX_VECTOR_CELLS
_MAX_VECTOR_ENTRY = (1 << 53) - 1


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"frames.{reason}", message)


class VectorFamily(StrictModel):
    """A bounded family of integer vectors in Euclidean coordinate space.

    Spanning and nonzero-vector requirements are operation-specific admission
    decisions made by the native kernels.
    """

    dimension: int = Field(ge=0, le=MAX_DIM)
    vectors: tuple[tuple[int, ...], ...] = Field(
        max_length=MAX_VECTOR_CELLS,
        description=(
            "Ordered vectors with len(vectors) * dimension <= "
            f"{MAX_VECTOR_CELLS} materialized cells."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        dimension = self.dimension
        if any(len(vector) != dimension for vector in self.vectors):
            raise _validation_error(
                "vector_dimension_mismatch", "all vectors must have equal dimension"
            )
        if len(self.vectors) * dimension > MAX_VECTOR_CELLS:
            raise _validation_error(
                "vector_cell_budget",
                "vector family exceeds the materialized-cell budget",
            )
        if any(
            abs(entry) > _MAX_VECTOR_ENTRY
            for vector in self.vectors
            for entry in vector
        ):
            raise _validation_error(
                "vector_entry_out_of_range", "vector entries must be bounded"
            )
        return self


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "VectorFamily",
]
