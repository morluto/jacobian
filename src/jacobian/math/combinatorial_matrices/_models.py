"""Typed wire contracts for combinatorial-matrix operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorial_matrices.values import HadamardMatrix, SignMatrix


class SignProfileRequest(StrictModel):
    """Compute the sign profile of a sign matrix."""

    matrix: SignMatrix


class SignProfileResult(StrictModel):
    """Dimensions, entry counts, row/column sums, and square-ness."""

    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    plus_one_count: int = Field(ge=0)
    minus_one_count: int = Field(ge=0)
    row_sums: tuple[int, ...]
    column_sums: tuple[int, ...]
    is_square: bool


class GramProfileRequest(StrictModel):
    """Compute the Gram profile of a sign matrix."""

    matrix: SignMatrix


class GramProfileResult(StrictModel):
    """Order, exact H H^T, diagonal residuals, nonzero off-diagonals, is_hadamard."""

    order: int = Field(ge=1)
    gram: tuple[tuple[int, ...], ...]
    diagonal_residuals: tuple[int, ...]
    nonzero_off_diagonal: tuple[tuple[int, int, int], ...]
    is_hadamard: bool


class NormalizeRequest(StrictModel):
    """Normalize a sign matrix so first row/column are all +1."""

    matrix: HadamardMatrix | SignMatrix


class NormalizeResult(StrictModel):
    """The normalized matrix and row/column sign switches used."""

    normalized: HadamardMatrix | SignMatrix
    row_switches: tuple[int, ...]
    column_switches: tuple[int, ...]

    @model_validator(mode="after")
    def bind_normalize(self) -> Self:
        for row in self.normalized.rows:
            for entry in row:
                if entry not in (-1, 1):
                    raise ValueError("normalized entries must be -1 or +1")
        return self


class DeterminantProfileRequest(StrictModel):
    """Compute the determinant profile of a Hadamard matrix."""

    matrix: HadamardMatrix


class DeterminantProfileResult(StrictModel):
    """Order, |det H|, Gram determinant, and the identity."""

    order: int = Field(ge=1)
    determinant_magnitude: int = Field(ge=1)
    gram_determinant: int = Field(ge=1)
    identity: str


class SylvesterRequest(StrictModel):
    """Construct the Sylvester Hadamard matrix of order 2^k."""

    k: int = Field(ge=0, le=7)


class SylvesterResult(StrictModel):
    """The constructed Sylvester matrix, construction name, and order."""

    matrix: HadamardMatrix
    construction: str
    order: int = Field(ge=1)


__all__ = [
    "DeterminantProfileRequest",
    "DeterminantProfileResult",
    "GramProfileRequest",
    "GramProfileResult",
    "NormalizeRequest",
    "NormalizeResult",
    "SignProfileRequest",
    "SignProfileResult",
    "SylvesterRequest",
    "SylvesterResult",
]
