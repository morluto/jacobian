"""Typed wire contracts for Latin square operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"combinatorics.latin_square.{reason}", message)


MAX_LATIN_SQUARE_ORDER = 1_024


def _validate_square_matrix(
    order: int,
    cells: tuple[tuple[int, ...], ...],
) -> None:
    """Validate that cells form an order x order matrix with values 0..order-1."""
    if len(cells) != order:
        raise _validation_error("row_count", "cells must be order x order")
    if any(len(row) != order for row in cells):
        raise _validation_error("row_length", "cells must be a square matrix")
    if any(not 0 <= v < order for row in cells for v in row):
        raise _validation_error("cell_range", "cell values must be in 0..order-1")


def _validate_latin_property(
    order: int,
    cells: tuple[tuple[int, ...], ...],
) -> None:
    """Validate that each symbol 0..order-1 appears exactly once per row and column."""
    if order == 0:
        return
    expected = set(range(order))
    for i in range(order):
        if set(cells[i]) != expected:
            raise _validation_error(
                "row_symbols", f"row {i} does not contain each symbol exactly once"
            )
    for j in range(order):
        if {cells[i][j] for i in range(order)} != expected:
            raise _validation_error(
                "column_symbols",
                f"column {j} does not contain each symbol exactly once",
            )


class LatinSquare(StrictModel):
    """A validated Latin square: an n x n matrix of symbols 0..n-1
    where each symbol appears exactly once in every row and column."""

    order: int = Field(ge=1, le=MAX_LATIN_SQUARE_ORDER)
    cells: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_LATIN_SQUARE_ORDER
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_square_matrix(self.order, self.cells)
        _validate_latin_property(self.order, self.cells)
        return self


class LatinSquareCandidate(StrictModel):
    """A candidate square matrix of symbols 0..n-1, not yet verified Latin."""

    order: int = Field(ge=1, le=MAX_LATIN_SQUARE_ORDER)
    cells: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_LATIN_SQUARE_ORDER
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_square_matrix(self.order, self.cells)
        return self


class LatinSquareRequest(StrictModel):
    """Check whether a candidate matrix is a Latin square."""

    square: LatinSquareCandidate


class TransposeRequest(StrictModel):
    """Transpose a validated Latin square."""

    square: LatinSquare


class OrthogonalityRequest(StrictModel):
    square_a: LatinSquare
    square_b: LatinSquare

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.square_a.order != self.square_b.order:
            raise _validation_error("order", "squares must have the same order")
        return self


# Results


class LatinSquareCheckResult(StrictModel):
    is_latin: bool


class OrthogonalityResult(StrictModel):
    is_orthogonal: bool
    pair_count: int = Field(ge=0)


class LatinSquareTransposeResult(StrictModel):
    transposed: LatinSquare

    @classmethod
    def _from_kernel(cls, *, order: int, cells: tuple[tuple[int, ...], ...]) -> Self:
        return cls.model_construct(
            transposed=LatinSquare.model_construct(order=order, cells=cells)
        )
