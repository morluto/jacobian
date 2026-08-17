"""Typed wire contracts for combinatorial matrix operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_HADAMARD_ORDER = 64


class MatrixEntry(StrictModel):
    """A single entry of a bounded integer matrix."""

    value: int


class HadamardCheckRequest(StrictModel):
    """Request a Hadamard-matrix check on a bounded square ±1 matrix."""

    matrix: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_HADAMARD_ORDER,
    )

    @model_validator(mode="after")
    def require_bounded_square_pm1_matrix(self) -> Self:
        order = len(self.matrix)
        if order < 1 or order > MAX_HADAMARD_ORDER:
            raise ValueError("matrix order must be between 1 and 64")
        for row in self.matrix:
            if len(row) != order:
                raise ValueError("matrix must be square")
            if any(entry not in (-1, 1) for entry in row):
                raise ValueError("matrix entries must be ±1")
        return self


class HadamardCheckResult(StrictModel):
    is_hadamard: bool
    order: int = Field(ge=1, le=MAX_HADAMARD_ORDER)


class HadamardNormalizeRequest(StrictModel):
    """Request normalization of a bounded square ±1 matrix."""

    matrix: tuple[tuple[int, ...], ...] = Field(
        min_length=1,
        max_length=MAX_HADAMARD_ORDER,
    )

    @model_validator(mode="after")
    def require_bounded_square_pm1_matrix(self) -> Self:
        order = len(self.matrix)
        if order < 1 or order > MAX_HADAMARD_ORDER:
            raise ValueError("matrix order must be between 1 and 64")
        for row in self.matrix:
            if len(row) != order:
                raise ValueError("matrix must be square")
            if any(entry not in (-1, 1) for entry in row):
                raise ValueError("matrix entries must be ±1")
        return self


class HadamardNormalizeResult(StrictModel):
    normalized_matrix: tuple[tuple[int, ...], ...]


class HadamardSylvesterRequest(StrictModel):
    """Request a Sylvester Hadamard matrix H_{2^k}."""

    k: int = Field(ge=0, le=6)


class HadamardSylvesterResult(StrictModel):
    matrix: tuple[tuple[int, ...], ...]
    order: int = Field(ge=1, le=MAX_HADAMARD_ORDER)
