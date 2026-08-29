"""Typed contracts for the axis-aligned square grid hypergraph operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_SIDE_LENGTH = 9


class AxisAlignedSquareGridRequest(StrictModel):
    """Request to construct the axis-aligned-square hypergraph of [N]^2."""

    side_length: int = Field(ge=1, le=MAX_SIDE_LENGTH)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        vertices = self.side_length**2
        if vertices > 256:
            raise PydanticCustomError(
                "square_grid.too_many_vertices",
                f"N^2 = {vertices} exceeds the 256-vertex limit",
            )
        return self


class AxisAlignedSquareGridResult(StrictModel):
    """The axis-aligned-square hypergraph of [N]^2."""

    side_length: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_SIDE_LENGTH",
    "AxisAlignedSquareGridRequest",
    "AxisAlignedSquareGridResult",
]
