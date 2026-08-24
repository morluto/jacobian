"""Typed wire contracts for matrix analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json

# The inertia result echoes its retained source matrix, so a request near
# the canonical input limit can produce a response past the identical
# output limit. Admission reserves this much for the inertia counts,
# definiteness label, and operation envelope beyond the echoed matrix.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


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
        output_limit = CanonicalLimits().max_output_bytes
        retained_bytes = len(encode_strict_json(self.model_dump(mode="json")))
        if retained_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
            raise ValueError(
                "the inertia result retains its source matrix and would "
                f"exceed the {output_limit}-byte canonical output limit; "
                "use fewer or smaller-magnitude entries"
            )
        return self


class InertiaResult(StrictModel):
    """Sylvester inertia (n_pos, n_neg, n_zero) of a symmetric matrix.

    Retains the canonical symmetric rational source matrix so validation
    replays the exact congruence-diagonal counts and enforces the
    definiteness label against them:

    - ``n_positive + n_negative + n_zero`` equals the dimension;
    - positive_definite iff all eigenvalues are positive, negative_definite
      iff all negative;
    - semidefinite labels require one zero sign class and none of the
      opposite sign; indefinite requires both nonzero sign classes.
    """

    matrix: SymmetricMatrixRequest
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    n_zero: int = Field(ge=0)
    definiteness: str

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.matrices.analysis._operations import (
            _build_matrix,
            _definiteness_label,
            _symmetric_inertia,
        )

        if self.n_positive + self.n_negative + self.n_zero != self.matrix.dimension:
            raise ValueError("inertia counts must sum to the matrix dimension")
        replayed = _symmetric_inertia(_build_matrix(self.matrix))
        if replayed != (self.n_positive, self.n_negative, self.n_zero):
            raise ValueError(
                "inertia counts must be the exact Sylvester inertia of the "
                "retained source matrix"
            )
        expected_label = _definiteness_label(
            self.n_positive, self.n_negative, self.n_zero
        )
        if self.definiteness != expected_label:
            raise ValueError(
                f"definiteness label must agree with the counts; expected "
                f"{expected_label!r}"
            )
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
