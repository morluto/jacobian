"""Typed wire contracts for matrix analysis operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions

# The canonical dense rational matrix carries determinant inputs through
# order 64, but the symmetric definiteness request's work and result
# budgets were established only through order 50. Pin the admitted
# dimension to that established envelope.
MAX_SYMMETRIC_MATRIX_DIMENSION = 50

# The inertia result echoes its source matrix in the domain's dense
# canonical form, so a request whose normalized echo is near the canonical
# output limit can produce a response past the identical limit. Admission
# reserves this much for the inertia counts, definiteness label, and
# operation envelope beyond the echoed matrix.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024
_RATIONAL_SPECTRUM_RESULT_BASE_BYTES = 2_048
_RATIONAL_SPECTRUM_MATRIX_ENTRY_BYTES = 96
_RATIONAL_SPECTRUM_CLAIM_BYTES = 256

MAX_RATIONAL_SPECTRUM_ORDER = 32
MAX_RATIONAL_SPECTRUM_CLAIMS = 32
MAX_RATIONAL_SPECTRUM_INPUT_DIGITS = 64
MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES = 1_024
MAX_RATIONAL_SPECTRUM_SHIFTED_DIGITS = 129
MAX_RATIONAL_SPECTRUM_RANK_WORK = 1_048_576
MAX_RATIONAL_SPECTRUM_MINOR_DIGITS = 132_256
MAX_RATIONAL_SPECTRUM_RESULT_BYTES = 384 * 1024


class RationalSpectrumMultiplicityClaim(StrictModel):
    """One distinct rational eigenvalue and its claimed positive multiplicity."""

    eigenvalue: CanonicalRational
    multiplicity: StrictInt = Field(ge=1, le=MAX_RATIONAL_SPECTRUM_ORDER)


class RationalSpectrumClaimRequest(StrictModel):
    """A complete rational spectrum claim for one symmetric matrix over QQ.

    ``matrix`` is the canonical materialized rational-matrix value. It must be
    square, symmetric, and have order at most 32. ``claimed_profile`` contains
    1..32 pairwise-distinct rational eigenvalues with positive multiplicities.
    Matrix and claim rational components are limited to 64 decimal digits.
    """

    matrix: RationalMatrix = Field(
        description=(
            "Canonical materialized square symmetric matrix over QQ, of order "
            "at most 32, with rational components of at most 64 decimal digits."
        )
    )
    claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_SPECTRUM_CLAIMS,
        description=(
            "Nonempty list of at most 32 pairwise-distinct rational eigenvalues "
            "and positive claimed multiplicities."
        ),
    )


class RationalSpectrumNullityLedgerEntry(StrictModel):
    """Exact shifted nullity and multiplicity comparison for one claim."""

    eigenvalue: CanonicalRational
    claimed_multiplicity: StrictInt = Field(ge=1, le=MAX_RATIONAL_SPECTRUM_ORDER)
    exact_nullity: StrictInt = Field(ge=0, le=MAX_RATIONAL_SPECTRUM_ORDER)
    multiplicity_matches: bool


RationalSpectrumFailure = Literal[
    "MULTIPLICITY_MISMATCH",
    "CLAIMED_MULTIPLICITY_SUM_DOES_NOT_EQUAL_MATRIX_ORDER",
]


class RationalSpectrumClaimResult(StrictModel):
    """An exact complete-spectrum decision bound to its source and claim."""

    matrix: RationalMatrix
    claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...] = Field(
        min_length=1, max_length=MAX_RATIONAL_SPECTRUM_CLAIMS
    )
    nullity_ledger: tuple[RationalSpectrumNullityLedgerEntry, ...] = Field(
        min_length=1, max_length=MAX_RATIONAL_SPECTRUM_CLAIMS
    )
    matrix_order: StrictInt = Field(ge=1, le=MAX_RATIONAL_SPECTRUM_ORDER)
    claimed_multiplicity_sum: StrictInt = Field(
        ge=1,
        le=MAX_RATIONAL_SPECTRUM_CLAIMS * MAX_RATIONAL_SPECTRUM_ORDER,
    )
    established_multiplicity_sum: StrictInt = Field(
        ge=0, le=MAX_RATIONAL_SPECTRUM_ORDER
    )
    outcome: Literal["VALID", "INVALID"]
    valid_complete_rational_spectrum: bool
    first_failed_condition: RationalSpectrumFailure | None = None
    first_failed_claim_index: StrictInt | None = Field(
        default=None, ge=0, lt=MAX_RATIONAL_SPECTRUM_CLAIMS
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        claimed_profile: tuple[RationalSpectrumMultiplicityClaim, ...],
        nullity_ledger: tuple[RationalSpectrumNullityLedgerEntry, ...],
        claimed_multiplicity_sum: int,
        established_multiplicity_sum: int,
        first_failed_claim_index: int | None,
        first_failed_condition: RationalSpectrumFailure | None,
    ) -> Self:
        """Construct a result emitted by the exact owner-local kernel."""

        valid = first_failed_condition is None
        return cls.model_construct(
            matrix=matrix,
            claimed_profile=claimed_profile,
            nullity_ledger=nullity_ledger,
            matrix_order=len(matrix.entries),
            claimed_multiplicity_sum=claimed_multiplicity_sum,
            established_multiplicity_sum=established_multiplicity_sum,
            outcome="VALID" if valid else "INVALID",
            valid_complete_rational_spectrum=valid,
            first_failed_condition=first_failed_condition,
            first_failed_claim_index=first_failed_claim_index,
        )


class MatrixEntry(StrictModel):
    """One rational matrix entry at (row, col)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: CanonicalRational


class SymmetricMatrixRequest(StrictModel):
    """A symmetric rational matrix for definiteness analysis."""

    dimension: int = Field(ge=1, le=MAX_SYMMETRIC_MATRIX_DIMENSION)
    entries: tuple[MatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for e in self.entries:
            if e.row >= self.dimension or e.col >= self.dimension:
                raise _validation_error(
                    "shape_mismatch", "entry indices must be < dimension"
                )
            key = (min(e.row, e.col), max(e.row, e.col))
            if key in seen:
                raise _validation_error(
                    "invariant_mismatch", "symmetric matrix entries must not conflict"
                )
            seen.add(key)
        return self


def _canonical_source_matrix(request: SymmetricMatrixRequest) -> RationalMatrix:
    """Normalize sparse symmetric input without entering the operation module."""

    matrix = [[Fraction(0)] * request.dimension for _ in range(request.dimension)]
    for entry in request.entries:
        value = entry.value.as_fraction()
        matrix[entry.row][entry.col] = value
        if entry.row != entry.col:
            matrix[entry.col][entry.row] = value
    return rational_matrix_from_fractions(matrix)


class InertiaResult(StrictModel):
    """Sylvester inertia (n_pos, n_neg, n_zero) of a symmetric matrix.

    Retains the source matrix in the domain's canonical dense
    ``RationalMatrix`` form, so every payload describing the same symmetric
    matrix yields identical outputs and digests regardless of entry order,
    triangular coordinates, or explicit zeros. Structural validation enforces
    the count and definiteness-label invariants. Exact congruence replay for
    independently supplied outcomes is provided by the owner verifier:

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
        dimension = len(self.matrix.entries)
        if dimension > MAX_SYMMETRIC_MATRIX_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                "retained source matrix exceeds the inertia order envelope",
            )
        if any(len(row) != dimension for row in self.matrix.entries):
            raise _validation_error(
                "shape_mismatch", "retained source matrix must be square"
            )
        if any(
            self.matrix.entries[row][col] != self.matrix.entries[col][row]
            for row in range(dimension)
            for col in range(row + 1, dimension)
        ):
            raise _validation_error(
                "shape_mismatch", "retained source matrix must be symmetric"
            )
        if self.n_positive + self.n_negative + self.n_zero != dimension:
            raise _validation_error(
                "shape_mismatch", "inertia counts must sum to the matrix dimension"
            )
        expected_label = _inertia_definiteness_label(
            self.n_positive, self.n_negative, self.n_zero
        )
        if self.definiteness != expected_label:
            raise _validation_error(
                "shape_mismatch",
                f"definiteness label must agree with the counts; expected "
                f"{expected_label!r}",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        n_positive: int,
        n_negative: int,
        n_zero: int,
    ) -> Self:
        """Construct a result emitted by the exact owner-local kernel."""

        return cls.model_construct(
            matrix=matrix,
            n_positive=n_positive,
            n_negative=n_negative,
            n_zero=n_zero,
            definiteness=_inertia_definiteness_label(n_positive, n_negative, n_zero),
        )


def _inertia_definiteness_label(n_pos: int, n_neg: int, n_zero: int) -> str:
    """Return the public definiteness label implied by one inertia triple."""

    if n_zero == 0:
        if n_neg == 0:
            return "positive_definite"
        if n_pos == 0:
            return "negative_definite"
        return "indefinite"
    if n_neg == 0:
        return "positive_semidefinite"
    if n_pos == 0:
        return "negative_semidefinite"
    return "indefinite"


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
            raise _validation_error(
                "shape_mismatch", "rhs_vector length must match constraint count"
            )
        if len(self.multipliers) != n_constraints:
            raise _validation_error(
                "shape_mismatch", "multipliers length must match constraint count"
            )
        widths = {len(row) for row in self.constraint_matrix}
        if len(widths) != 1 or 0 in widths:
            raise _validation_error(
                "shape_mismatch",
                "constraint matrix must be rectangular with positive row width",
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
    "RationalSpectrumClaimRequest",
    "RationalSpectrumClaimResult",
    "RationalSpectrumMultiplicityClaim",
    "RationalSpectrumNullityLedgerEntry",
    "SymmetricMatrixRequest",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
