"""Typed wire contracts for matrix analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION, RationalMatrix

# The inertia result echoes its source matrix in the domain's dense
# canonical form, so a request whose normalized echo is near the canonical
# output limit can produce a response past the identical limit. Admission
# reserves this much for the inertia counts, definiteness label, and
# operation envelope beyond the echoed matrix.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


class MatrixEntry(StrictModel):
    """One rational matrix entry at (row, col)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: CanonicalRational


class SymmetricMatrixRequest(StrictModel):
    """A symmetric rational matrix for definiteness analysis."""

    dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    entries: tuple[MatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from jacobian.math.matrices.analysis._operations import (
            _canonical_source_matrix,
        )

        seen: set[tuple[int, int]] = set()
        for e in self.entries:
            if e.row >= self.dimension or e.col >= self.dimension:
                raise ValueError("entry indices must be < dimension")
            key = (min(e.row, e.col), max(e.row, e.col))
            if key in seen:
                raise ValueError("symmetric matrix entries must not conflict")
            seen.add(key)
        source = _canonical_source_matrix(self)
        output_limit = CanonicalLimits().max_output_bytes
        try:
            retained_bytes = len(
                encode_strict_json(source.model_dump(mode="json"))
            )
        except CanonicalizationError:
            retained_bytes = output_limit + 1
        if retained_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
            raise ValueError(
                "the inertia result retains its source matrix and would "
                f"exceed the {output_limit}-byte canonical output limit; "
                "use fewer or smaller-magnitude entries"
            )
        return self


class InertiaResult(StrictModel):
    """Sylvester inertia (n_pos, n_neg, n_zero) of a symmetric matrix.

    Retains the source matrix in the domain's canonical dense
    ``RationalMatrix`` form, so every payload describing the same symmetric
    matrix yields identical outputs and digests regardless of entry order,
    triangular coordinates, or explicit zeros. Validation replays the exact
    congruence-diagonal counts and enforces the definiteness label against
    them:

    - ``n_positive + n_negative + n_zero`` equals the dimension;
    - positive_definite iff all eigenvalues are positive, negative_definite
      iff all negative;
    - semidefinite labels require one zero sign class and none of the
      opposite sign; indefinite requires both nonzero sign classes.
    """

    matrix: RationalMatrix
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    n_zero: int = Field(ge=0)
    definiteness: str

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.matrices.analysis._operations import (
            _definiteness_label,
            _dense_fractions,
            _symmetric_inertia,
        )

        dimension = len(self.matrix.entries)
        if any(len(row) != dimension for row in self.matrix.entries):
            raise ValueError("retained source matrix must be square")
        if any(
            self.matrix.entries[row][col] != self.matrix.entries[col][row]
            for row in range(dimension)
            for col in range(row + 1, dimension)
        ):
            raise ValueError("retained source matrix must be symmetric")
        if self.n_positive + self.n_negative + self.n_zero != dimension:
            raise ValueError("inertia counts must sum to the matrix dimension")
        replayed = _symmetric_inertia(_dense_fractions(self.matrix))
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
