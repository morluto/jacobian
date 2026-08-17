"""Typed wire contracts for matrix analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class MatrixEntry(StrictModel):
    """One rational matrix entry at (row, col)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: CanonicalRational


class SymmetricMatrixRequest(StrictModel):
    """A symmetric rational matrix for definiteness analysis."""

    dimension: int = Field(ge=1, le=50)
    entries: tuple[MatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for e in self.entries:
            if e.row >= self.dimension or e.col >= self.dimension:
                raise ValueError("entry indices must be < dimension")
            key = (min(e.row, e.col), max(e.row, e.col))
            if key in seen:
                raise ValueError("symmetric matrix entries must not conflict")
            seen.add(key)
        return self


class InertiaResult(StrictModel):
    """Sylvester inertia (n_pos, n_neg, n_zero) of a symmetric matrix."""

    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    n_zero: int = Field(ge=0)
    definiteness: str

    @model_validator(mode="after")
    def require_dimension(self) -> Self:
        if self.definiteness not in (
            "positive_definite",
            "positive_semidefinite",
            "negative_definite",
            "negative_semidefinite",
            "indefinite",
        ):
            raise ValueError("invalid definiteness label")
        return self


class FarkasCertificateRequest(StrictModel):
    """Check a rational Farkas infeasibility certificate.

    Given Ax <= b and a non-negative multiplier vector y, verify that
    y^T A = 0 and y^T b < 0, proving the system is infeasible.
    """

    constraint_matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(min_length=1)
    rhs_vector: tuple[CanonicalRational, ...] = Field(min_length=1)
    multipliers: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n_constraints = len(self.constraint_matrix)
        if len(self.rhs_vector) != n_constraints:
            raise ValueError("rhs_vector length must match constraint count")
        if len(self.multipliers) != n_constraints:
            raise ValueError("multipliers length must match constraint count")
        widths = {len(row) for row in self.constraint_matrix}
        if len(widths) != 1 or 0 in widths:
            raise ValueError(
                "constraint matrix must be rectangular with positive row width"
            )
        return self


class FarkasCertificateResult(StrictModel):
    """Result of checking a Farkas infeasibility certificate."""

    valid: bool
    y_t_a: tuple[str, ...]
    y_t_b: str
    reason: str


__all__ = [
    "FarkasCertificateRequest",
    "FarkasCertificateResult",
    "InertiaResult",
    "MatrixEntry",
    "SymmetricMatrixRequest",
]
