"""Typed wire contracts for cubical complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_DIM = 10
MAX_CELLS = 5000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by cubical-complex contracts."""

    return PydanticCustomError(f"cubical_complex.{reason}", message)


class CubicalCell(StrictModel):
    """An elementary cube: a tuple of intervals [a_i, b_i] on integer lattice."""

    intervals: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid_intervals(self) -> Self:
        for a, b in self.intervals:
            if a > b:
                raise _validation_error(
                    "interval_order",
                    "each interval must have a <= b (interval is [a, b])",
                )
            if b - a > 1:
                raise _validation_error(
                    "interval_length",
                    "each interval must have length 0 or 1 (b <= a + 1)",
                )
        return self

    @property
    def dimension(self) -> int:
        return sum(1 for a, b in self.intervals if b > a)


class CubicalComplexRequest(StrictModel):
    """A finite cubical complex: a set of elementary cubes."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class CubicalComplex(StrictModel):
    """Canonical cubical complex with an explicit ambient coordinate axis.

    ``cells`` is a sorted family of distinct cells.  Operations establish that
    it is face closed before constructing this value; decoding checks only the
    bounded cell and axis representation.
    """

    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)

    @model_validator(mode="after")
    def require_structural_cells(self) -> Self:
        if any(len(cell.intervals) != self.ambient_dimension for cell in self.cells):
            raise _validation_error(
                "ambient_dimension_mismatch",
                "every cell must use the declared ambient coordinate axis",
            )
        if tuple(sorted(self.cells, key=lambda cell: cell.intervals)) != self.cells:
            raise _validation_error(
                "cells_not_canonical", "cells must be sorted canonically"
            )
        if len(set(self.cells)) != len(self.cells):
            raise _validation_error("duplicate_cells", "cells must be distinct")
        return self


class FVector(StrictModel):
    """An f-vector whose entries are indexed by explicit cell dimension."""

    dimension_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)
    counts: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        if self.dimension_axis != tuple(range(len(self.dimension_axis))):
            raise _validation_error(
                "dimension_axis_not_canonical",
                "dimension axis must enumerate dimensions from zero",
            )
        if len(self.counts) != len(self.dimension_axis) or any(
            count < 0 for count in self.counts
        ):
            raise _validation_error(
                "f_vector_shape", "f-vector counts must match its dimension axis"
            )
        return self


class FVectorResult(StrictModel):
    """The f-vector and Euler characteristic bound to a cubical complex."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    f_vector: FVector
    euler_characteristic: int


class FaceClosureRequest(StrictModel):
    """Compute the full face closure of a set of cells."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class FaceClosureResult(StrictModel):
    """A face-closed complex and its dimensional cell-count summary."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    original_cells: int
    total_cells: int
    cells_by_dimension: FVector


__all__ = [
    "MAX_CELLS",
    "MAX_DIM",
    "CubicalCell",
    "CubicalComplex",
    "CubicalComplexRequest",
    "FVector",
    "FVectorResult",
    "FaceClosureRequest",
    "FaceClosureResult",
]
