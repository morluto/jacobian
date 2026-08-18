"""Typed wire contracts for Latin square operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_N = 32


class LatinSquare(StrictModel):
    """A Latin square as an n x n matrix of symbols 0..n-1."""

    order: int = Field(ge=1, le=MAX_N)
    cells: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_N)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.cells) != self.order:
            raise ValueError("cells must be order x order")
        if any(len(row) != self.order for row in self.cells):
            raise ValueError("cells must be a square matrix")
        if any(not 0 <= v < self.order for row in self.cells for v in row):
            raise ValueError("cell values must be in 0..order-1")
        return self


class LatinSquareRequest(StrictModel):
    square: LatinSquare


class OrthogonalityRequest(StrictModel):
    square_a: LatinSquare
    square_b: LatinSquare

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.square_a.order != self.square_b.order:
            raise ValueError("squares must have the same order")
        return self


# Results


class LatinSquareCheckResult(StrictModel):
    is_latin: bool
    method: str = "ROW_COLUMN_SYMBOL_UNIQUENESS"


class OrthogonalityResult(StrictModel):
    is_orthogonal: bool
    pair_count: int = Field(ge=0)
    method: str = "ORDERED_PAIR_UNIQUENESS"


class LatinSquareTransposeResult(StrictModel):
    transposed: tuple[tuple[int, ...], ...]
    method: str = "MATRIX_TRANSPOSE"
