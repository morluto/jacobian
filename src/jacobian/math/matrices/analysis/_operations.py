"""Domain-owned matrix analysis operations."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from jacobian._exact import CanonicalRational, format_canonical_rational
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    FarkasCertificateResult,
    InertiaResult,
    RationalSpectrumClaimRequest,
    RationalSpectrumClaimResult,
    RationalSpectrumFailure,
    RationalSpectrumNullityLedgerEntry,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


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
    ledger, claimed_sum, established_sum, mismatch, failure = (
        _replay_rational_spectrum_claim(request)
    )
    valid = failure is None
    return RationalSpectrumClaimResult(
        matrix=request.matrix,
        claimed_profile=request.claimed_profile,
        nullity_ledger=ledger,
        matrix_order=len(request.matrix.entries),
        claimed_multiplicity_sum=claimed_sum,
        established_multiplicity_sum=established_sum,
        outcome="VALID" if valid else "INVALID",
        valid_complete_rational_spectrum=valid,
        first_failed_condition=failure,
        first_failed_claim_index=mismatch,
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


def _canonical_source_matrix(request: SymmetricMatrixRequest) -> RationalMatrix:
    """Normalize the sparse symmetric request into the canonical dense value."""
    return rational_matrix_from_fractions(_build_matrix(request))


def _dense_fractions(matrix: RationalMatrix) -> list[list[Fraction]]:
    """Convert a canonical rational matrix into dense Fractions."""
    return [[entry.as_fraction() for entry in row] for row in matrix.entries]


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


def _definiteness_label(n_pos: int, n_neg: int, n_zero: int) -> str:
    """Return the definiteness label implied by one inertia triple."""
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


def compute_inertia(request: SymmetricMatrixRequest) -> InertiaResult:
    """Compute the Sylvester inertia of a symmetric rational matrix."""
    n_pos, n_neg, n_zero = _symmetric_inertia(_build_matrix(request))
    return InertiaResult(
        matrix=_canonical_source_matrix(request),
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
        definiteness=_definiteness_label(n_pos, n_neg, n_zero),
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
