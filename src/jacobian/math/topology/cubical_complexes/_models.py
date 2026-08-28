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


class FVectorResult(StrictModel):
    """The f-vector and Euler characteristic of a cubical complex."""

    dimension: int
    f_vector: tuple[int, ...]
    euler_characteristic: int


class FaceClosureRequest(StrictModel):
    """Compute the full face closure of a set of cells."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class FaceClosureResult(StrictModel):
    """Result of face closure computation."""

    original_cells: int
    total_cells: int
    cells_by_dimension: tuple[int, ...]
