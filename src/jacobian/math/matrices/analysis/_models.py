"""Typed wire contracts for matrix analysis operations."""

from __future__ import annotations

from math import factorial
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    RationalMatrix,
    require_matrix_scalar_digits,
)

# The inertia result echoes its source matrix in the domain's dense
# canonical form, so a request whose normalized echo is near the canonical
# output limit can produce a response past the identical limit. Admission
# reserves this much for the inertia counts, definiteness label, and
# operation envelope beyond the echoed matrix.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024

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

    @model_validator(mode="after")
    def require_admitted_complete_claim(self) -> Self:
        order = len(self.matrix.entries)
        if order != len(self.matrix.entries[0]):
            raise _validation_error(
                "shape_mismatch", "rational spectrum claims require a square matrix"
            )
        if order > MAX_RATIONAL_SPECTRUM_ORDER:
            raise _validation_error(
                "budget_exceeded",
                "rational spectrum claims support matrix order at most "
                f"{MAX_RATIONAL_SPECTRUM_ORDER}",
            )
        if any(
            self.matrix.entries[row][column] != self.matrix.entries[column][row]
            for row in range(order)
            for column in range(row + 1, order)
        ):
            raise _validation_error(
                "budget_exceeded", "rational spectrum claims require a symmetric matrix"
            )

        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
            label="rational spectrum matrix",
        )
        nonzero_entries = sum(
            entry.num != "0" for row in self.matrix.entries for entry in row
        )
        if nonzero_entries > MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES:
            raise _validation_error(
                "budget_exceeded",
                "rational spectrum matrix exceeds the nonzero-entry budget",
            )
        eigenvalues = tuple(claim.eigenvalue for claim in self.claimed_profile)
        for eigenvalue in eigenvalues:
            require_bounded_rational(
                eigenvalue,
                max_digits=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
                label="claimed eigenvalue",
            )
        if len(set(eigenvalues)) != len(eigenvalues):
            raise _validation_error(
                "budget_exceeded",
                "claimed rational eigenvalues must be pairwise distinct",
            )

        shifted_digits = max(
            canonical_rational_component_digits(
                CanonicalRational.from_fraction(
                    self.matrix.entries[index][index].as_fraction()
                    - eigenvalue.as_fraction()
                )
            )
            for eigenvalue in eigenvalues
            for index in range(order)
        )
        if shifted_digits > MAX_RATIONAL_SPECTRUM_SHIFTED_DIGITS:
            raise _validation_error(
                "budget_exceeded",
                "shifted diagonal entries exceed the rational spectrum digit budget",
            )

        rank_work = len(eigenvalues) * order**3
        if rank_work > MAX_RATIONAL_SPECTRUM_RANK_WORK:
            raise _validation_error(
                "budget_exceeded",
                "shifted-rank computations exceed the aggregate work budget",
            )

        # Clearing each row's denominators gives integer entries with at most
        # order * shifted_digits digits. Every square minor then has at most
        # order! terms, each a product of at most order such entries.
        minor_digits = order * order * shifted_digits + len(str(factorial(order))) + 1
        if minor_digits > MAX_RATIONAL_SPECTRUM_MINOR_DIGITS:
            raise _validation_error(
                "budget_exceeded", "exact shifted-rank minors exceed the digit budget"
            )

        source_cells = order * order
        result_bytes = (
            2_048
            + source_cells * (2 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS + 96)
            + len(eigenvalues) * (4 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS + 256)
        )
        if result_bytes > MAX_RATIONAL_SPECTRUM_RESULT_BYTES:
            raise _validation_error(
                "budget_exceeded",
                "rational spectrum ledger exceeds the result-size budget",
            )
        return self


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
    method: Literal["PYTHON_FLINT_EXACT_RANK_AFTER_ROW_DENOMINATOR_CLEARING"] = (
        "PYTHON_FLINT_EXACT_RANK_AFTER_ROW_DENOMINATOR_CLEARING"
    )

    @model_validator(mode="after")
    def bind_complete_claim_to_source(self) -> Self:
        request = RationalSpectrumClaimRequest(
            matrix=self.matrix,
            claimed_profile=self.claimed_profile,
        )
        from jacobian.math.matrices.analysis._operations import (
            _replay_rational_spectrum_claim,
        )

        (
            expected_ledger,
            claimed_sum,
            established_sum,
            mismatch,
            failure,
        ) = _replay_rational_spectrum_claim(request)
        valid = failure is None

        expected = (
            self.nullity_ledger == expected_ledger
            and self.matrix_order == len(request.matrix.entries)
            and self.claimed_multiplicity_sum == claimed_sum
            and self.established_multiplicity_sum == established_sum
            and self.outcome == ("VALID" if valid else "INVALID")
            and self.valid_complete_rational_spectrum is valid
            and self.first_failed_condition == failure
            and self.first_failed_claim_index == mismatch
        )
        if not expected:
            raise _validation_error(
                "shape_mismatch",
                "rational spectrum result does not match exact replay from the "
                "retained source and claim",
            )
        return self


class MatrixEntry(StrictModel):
    """One rational matrix entry at (row, col)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: CanonicalRational


class SymmetricMatrixRequest(StrictModel):
    """A symmetric rational matrix for definiteness analysis."""

    dimension: int = Field(ge=1, le=MAX_RATIONAL_MATRIX_ORDER)
    entries: tuple[MatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from jacobian.math.matrices.analysis._operations import (
            _canonical_source_matrix,
        )

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
        source = _canonical_source_matrix(self)
        output_limit = CanonicalLimits().max_output_bytes
        try:
            retained_bytes = len(encode_strict_json(source.model_dump(mode="json")))
        except CanonicalizationError:
            retained_bytes = output_limit + 1
        if retained_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
            raise _validation_error(
                "invariant_mismatch",
                "the inertia result retains its source matrix and would "
                f"exceed the {output_limit}-byte canonical output limit; "
                "use fewer or smaller-magnitude entries",
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
        replayed = _symmetric_inertia(_dense_fractions(self.matrix))
        if replayed != (self.n_positive, self.n_negative, self.n_zero):
            raise _validation_error(
                "shape_mismatch",
                "inertia counts must be the exact Sylvester inertia of the "
                "retained source matrix",
            )
        expected_label = _definiteness_label(
            self.n_positive, self.n_negative, self.n_zero
        )
        if self.definiteness != expected_label:
            raise _validation_error(
                "shape_mismatch",
                f"definiteness label must agree with the counts; expected "
                f"{expected_label!r}",
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
