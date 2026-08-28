"""Domain-owned matrix analysis operations."""

from __future__ import annotations

from fractions import Fraction
from math import factorial, lcm
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
    format_canonical_rational,
    require_bounded_rational,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.analysis._models import (
    _RATIONAL_SPECTRUM_CLAIM_BYTES,
    _RATIONAL_SPECTRUM_MATRIX_ENTRY_BYTES,
    _RATIONAL_SPECTRUM_RESULT_BASE_BYTES,
    _RESULT_ENVELOPE_RESERVE_BYTES,
    MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
    MAX_RATIONAL_SPECTRUM_MINOR_DIGITS,
    MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES,
    MAX_RATIONAL_SPECTRUM_ORDER,
    MAX_RATIONAL_SPECTRUM_RANK_WORK,
    MAX_RATIONAL_SPECTRUM_RESULT_BYTES,
    MAX_RATIONAL_SPECTRUM_SHIFTED_DIGITS,
    FarkasCertificateRequest,
    FarkasCertificateResult,
    InertiaResult,
    RationalSpectrumClaimRequest,
    RationalSpectrumClaimResult,
    RationalSpectrumFailure,
    RationalSpectrumNullityLedgerEntry,
    SymmetricMatrixRequest,
    _canonical_source_matrix,
    _validation_error,
)
from jacobian.math.matrices.values import RationalMatrix, require_matrix_scalar_digits


def _admit_rational_spectrum_claim(request: RationalSpectrumClaimRequest) -> None:
    order = len(request.matrix.entries)
    if order != len(request.matrix.entries[0]):
        raise _validation_error(
            "shape_mismatch", "rational spectrum claims require a square matrix"
        )
    if order > MAX_RATIONAL_SPECTRUM_ORDER:
        raise _validation_error(
            "budget_exceeded",
            f"rational spectrum claims support matrix order at most {MAX_RATIONAL_SPECTRUM_ORDER}",
        )
    if any(
        request.matrix.entries[row][column] != request.matrix.entries[column][row]
        for row in range(order)
        for column in range(row + 1, order)
    ):
        raise _validation_error(
            "budget_exceeded", "rational spectrum claims require a symmetric matrix"
        )
    require_matrix_scalar_digits(
        request.matrix.entries,
        maximum=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
        label="rational spectrum matrix",
    )
    nonzero_entries = sum(
        entry.num != "0" for row in request.matrix.entries for entry in row
    )
    if nonzero_entries > MAX_RATIONAL_SPECTRUM_NONZERO_ENTRIES:
        raise _validation_error(
            "budget_exceeded",
            "rational spectrum matrix exceeds the nonzero-entry budget",
        )
    eigenvalues = tuple(claim.eigenvalue for claim in request.claimed_profile)
    for eigenvalue in eigenvalues:
        require_bounded_rational(
            eigenvalue,
            max_digits=MAX_RATIONAL_SPECTRUM_INPUT_DIGITS,
            label="claimed eigenvalue",
        )
    if len(set(eigenvalues)) != len(eigenvalues):
        raise _validation_error(
            "budget_exceeded", "claimed rational eigenvalues must be pairwise distinct"
        )
    shifted_digits = max(
        canonical_rational_component_digits(
            CanonicalRational.from_fraction(
                request.matrix.entries[index][index].as_fraction()
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
    if len(eigenvalues) * order**3 > MAX_RATIONAL_SPECTRUM_RANK_WORK:
        raise _validation_error(
            "budget_exceeded",
            "shifted-rank computations exceed the aggregate work budget",
        )
    minor_digits = order * order * shifted_digits + len(str(factorial(order))) + 1
    if minor_digits > MAX_RATIONAL_SPECTRUM_MINOR_DIGITS:
        raise _validation_error(
            "budget_exceeded", "exact shifted-rank minors exceed the digit budget"
        )
    result_bytes = (
        _RATIONAL_SPECTRUM_RESULT_BASE_BYTES
        + order
        * order
        * (
            2 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS
            + _RATIONAL_SPECTRUM_MATRIX_ENTRY_BYTES
        )
        + len(eigenvalues)
        * (4 * MAX_RATIONAL_SPECTRUM_INPUT_DIGITS + _RATIONAL_SPECTRUM_CLAIM_BYTES)
    )
    if result_bytes > MAX_RATIONAL_SPECTRUM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded", "rational spectrum ledger exceeds the result-size budget"
        )


def _admit(request: Any) -> None:
    try:
        if isinstance(request, RationalSpectrumClaimRequest):
            _admit_rational_spectrum_claim(request)
        else:
            seen: set[tuple[int, int]] = set()
            for entry in request.entries:
                if entry.row >= request.dimension or entry.col >= request.dimension:
                    raise _validation_error(
                        "shape_mismatch", "entry indices must be < dimension"
                    )
                key = (min(entry.row, entry.col), max(entry.row, entry.col))
                if key in seen:
                    raise _validation_error(
                        "invariant_mismatch",
                        "symmetric matrix entries must not conflict",
                    )
                seen.add(key)
            source = _canonical_source_matrix(request)
            output_limit = CanonicalLimits().max_output_bytes
            try:
                retained_bytes = len(encode_strict_json(source.model_dump(mode="json")))
            except CanonicalizationError:
                retained_bytes = output_limit + 1
            if retained_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
                raise _validation_error(
                    "invariant_mismatch",
                    "the inertia result retains its source matrix and would exceed the canonical output limit",
                )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("request",), code=exc.type, message=exc.message()
        ) from exc


def _exact_shifted_nullities(
    matrix: RationalMatrix,
    eigenvalues: tuple[CanonicalRational, ...],
) -> tuple[int, ...]:
    """Return nullities of ``matrix - eigenvalue * I`` over QQ.

    Each rational row is scaled by the least common multiple of its
    denominators. This preserves rank and gives FLINT an exact integer matrix;
    no floating-point or symbolic expression crosses the adapter boundary.
    """
    from flint import fmpz_mat

    source = tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.entries
    )
    order = len(source)
    nullities: list[int] = []
    for canonical_eigenvalue in eigenvalues:
        eigenvalue = canonical_eigenvalue.as_fraction()
        integer_rows: list[list[int]] = []
        for row_index, source_row in enumerate(source):
            shifted_row = [
                entry - eigenvalue if row_index == column_index else entry
                for column_index, entry in enumerate(source_row)
            ]
            row_denominator = lcm(*(entry.denominator for entry in shifted_row))
            integer_rows.append(
                [
                    entry.numerator * (row_denominator // entry.denominator)
                    for entry in shifted_row
                ]
            )
        nullities.append(order - int(fmpz_mat(integer_rows).rank()))
    return tuple(nullities)


def check_rational_spectrum_claim(
    request: RationalSpectrumClaimRequest,
) -> RationalSpectrumClaimResult:
    """Check a claimed complete rational spectrum of a symmetric QQ matrix."""
    _admit(request)
    ledger, claimed_sum, established_sum, mismatch, failure = (
        _replay_rational_spectrum_claim(request)
    )
    return RationalSpectrumClaimResult._from_kernel(
        request=request,
        nullity_ledger=ledger,
        claimed_multiplicity_sum=claimed_sum,
        established_multiplicity_sum=established_sum,
        first_failed_claim_index=mismatch,
        first_failed_condition=failure,
    )


def _replay_rational_spectrum_claim(
    request: RationalSpectrumClaimRequest,
) -> tuple[
    tuple[RationalSpectrumNullityLedgerEntry, ...],
    int,
    int,
    int | None,
    RationalSpectrumFailure | None,
]:
    """Recompute the complete source-bound claim ledger."""
    nullities = _exact_shifted_nullities(
        request.matrix,
        tuple(claim.eigenvalue for claim in request.claimed_profile),
    )
    ledger = tuple(
        RationalSpectrumNullityLedgerEntry(
            eigenvalue=claim.eigenvalue,
            claimed_multiplicity=claim.multiplicity,
            exact_nullity=nullity,
            multiplicity_matches=nullity == claim.multiplicity,
        )
        for claim, nullity in zip(request.claimed_profile, nullities, strict=True)
    )
    claimed_sum = sum(claim.multiplicity for claim in request.claimed_profile)
    mismatch = next(
        (index for index, entry in enumerate(ledger) if not entry.multiplicity_matches),
        None,
    )
    if mismatch is not None:
        failure: RationalSpectrumFailure | None = "MULTIPLICITY_MISMATCH"
    elif claimed_sum != len(request.matrix.entries):
        failure = "CLAIMED_MULTIPLICITY_SUM_DOES_NOT_EQUAL_MATRIX_ORDER"
    else:
        failure = None
    return ledger, claimed_sum, sum(nullities), mismatch, failure


def _build_matrix(request: SymmetricMatrixRequest) -> list[list[Fraction]]:
    """Build a full symmetric matrix from sparse entries."""
    n = request.dimension
    mat = [[Fraction(0)] * n for _ in range(n)]
    for entry in request.entries:
        value = entry.value.as_fraction()
        mat[entry.row][entry.col] = value
        if entry.row != entry.col:
            mat[entry.col][entry.row] = value
    return mat


def _swap_symmetric(matrix: list[list[Fraction]], left: int, right: int) -> None:
    if left == right:
        return
    matrix[left], matrix[right] = matrix[right], matrix[left]
    for row in matrix:
        row[left], row[right] = row[right], row[left]


def _count_2x2_inertia(
    aa: Fraction, bb: Fraction, cc: Fraction
) -> tuple[int, int, int]:
    det = aa * cc - bb * bb
    trace = aa + cc
    if det < 0:
        return 1, 1, 0
    if det > 0:
        if aa > 0 or (aa == 0 and cc > 0):
            return 2, 0, 0
        return 0, 2, 0
    if trace > 0:
        return 1, 0, 1
    if trace < 0:
        return 0, 1, 1
    return 0, 0, 2


def _eliminate_1x1(matrix: list[list[Fraction]], index: int, pivot: int) -> int:
    _swap_symmetric(matrix, index, pivot)
    diagonal = matrix[index][index]
    for row in range(index + 1, len(matrix)):
        if matrix[row][index] == 0:
            continue
        factor = matrix[row][index] / diagonal
        for col in range(index, len(matrix)):
            matrix[row][col] -= factor * matrix[index][col]
        for col in range(index, len(matrix)):
            matrix[col][row] = matrix[row][col]
    return 1 if diagonal > 0 else -1


def _find_off_diagonal(
    matrix: list[list[Fraction]], index: int
) -> tuple[int, int] | None:
    for row in range(index, len(matrix)):
        for col in range(row + 1, len(matrix)):
            if matrix[row][col] != 0:
                return row, col
    return None


def _eliminate_2x2(matrix: list[list[Fraction]], index: int) -> tuple[int, int, int]:
    first, second = _find_off_diagonal(matrix, index) or (index, index)
    _swap_symmetric(matrix, index, first)
    if second == index:
        second = first
    _swap_symmetric(matrix, index + 1, second)
    pos, neg, zero = _count_2x2_inertia(
        matrix[index][index],
        matrix[index][index + 1],
        matrix[index + 1][index + 1],
    )
    det = (
        matrix[index][index] * matrix[index + 1][index + 1]
        - matrix[index][index + 1] ** 2
    )
    if index + 2 < len(matrix) and det != 0:
        inv00 = matrix[index + 1][index + 1] / det
        inv01 = -matrix[index][index + 1] / det
        inv11 = matrix[index][index] / det
        for row in range(index + 2, len(matrix)):
            left = matrix[row][index]
            right = matrix[row][index + 1]
            coeff0 = left * inv00 + right * inv01
            coeff1 = left * inv01 + right * inv11
            for col in range(index, len(matrix)):
                matrix[row][col] -= (
                    coeff0 * matrix[index][col] + coeff1 * matrix[index + 1][col]
                )
            for col in range(index, len(matrix)):
                matrix[col][row] = matrix[row][col]
    return pos, neg, zero


def _symmetric_inertia(matrix: list[list[Fraction]]) -> tuple[int, int, int]:
    """Reduce a symmetric rational matrix to a congruence-diagonal form."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    n_pos = n_neg = n_zero = 0
    index = 0
    while index < n:
        pivot = next((row for row in range(index, n) if a[row][row] != 0), None)
        if pivot is not None:
            sign = _eliminate_1x1(a, index, pivot)
            n_pos += sign > 0
            n_neg += sign < 0
            index += 1
            continue
        if _find_off_diagonal(a, index) is None:
            n_zero += n - index
            break
        pos, neg, zero = _eliminate_2x2(a, index)
        n_pos += pos
        n_neg += neg
        n_zero += zero
        index += 2
    return n_pos, n_neg, n_zero


def compute_inertia(request: SymmetricMatrixRequest) -> InertiaResult:
    """Compute the Sylvester inertia of a symmetric rational matrix."""
    _admit(request)
    n_pos, n_neg, n_zero = _symmetric_inertia(_build_matrix(request))
    return InertiaResult._from_kernel(
        matrix=_canonical_source_matrix(request),
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
    )


def check_farkas_certificate(
    request: FarkasCertificateRequest,
) -> FarkasCertificateResult:
    """Check a rational Farkas infeasibility certificate.

    Given system Ax <= b and multiplier vector y >= 0, the certificate is
    valid if y^T A = 0 and y^T b < 0.
    """
    y = [multiplier.as_fraction() for multiplier in request.multipliers]
    constraint_matrix = [
        [entry.as_fraction() for entry in row] for row in request.constraint_matrix
    ]
    b = [entry.as_fraction() for entry in request.rhs_vector]

    if any(entry < 0 for entry in y):
        ytb = sum((yi * bi for yi, bi in zip(y, b, strict=True)), Fraction(0))
        return FarkasCertificateResult(
            valid=False,
            y_t_a=(),
            y_t_b=format_canonical_rational(ytb),
            reason="multiplier vector has a negative entry",
        )

    n_vars = len(constraint_matrix[0])
    yta = [Fraction(0)] * n_vars
    for i, yi in enumerate(y):
        for j in range(n_vars):
            yta[j] += yi * constraint_matrix[i][j]
    ytb = sum((yi * bi for yi, bi in zip(y, b, strict=True)), Fraction(0))
    yta_str = tuple(format_canonical_rational(value) for value in yta)

    if all(value == 0 for value in yta) and ytb < 0:
        return FarkasCertificateResult(
            valid=True,
            y_t_a=yta_str,
            y_t_b=format_canonical_rational(ytb),
            reason="y^T A = 0 and y^T b < 0",
        )
    reasons = []
    if any(value != 0 for value in yta):
        reasons.append("y^T A != 0")
    if ytb >= 0:
        reasons.append("y^T b >= 0")
    return FarkasCertificateResult(
        valid=False,
        y_t_a=yta_str,
        y_t_b=format_canonical_rational(ytb),
        reason="; ".join(reasons) if reasons else "unknown",
    )


__all__ = [
    "check_farkas_certificate",
    "check_rational_spectrum_claim",
    "compute_inertia",
]
