"""Typed contracts for the axis-aligned square grid hypergraph operation."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_SIDE_LENGTH = 16


class AxisAlignedSquareGridRequest(StrictModel):
    """Request to construct the axis-aligned-square hypergraph of [N]^2."""

    side_length: StrictInt = Field(ge=1, le=MAX_SIDE_LENGTH)


class AxisAlignedSquareGridResult(StrictModel):
    """The axis-aligned-square hypergraph of [N]^2."""

    side_length: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_SIDE_LENGTH",
    "AxisAlignedSquareGridRequest",
    "AxisAlignedSquareGridResult",
]
