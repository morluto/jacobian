"""Domain-owned matrix analysis operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.math.matrix_analysis._models import (
    FarkasCertificateRequest,
    FarkasCertificateResult,
    InertiaResult,
    SymmetricMatrixRequest,
)


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
    n_pos, n_neg, n_zero = _symmetric_inertia(_build_matrix(request))
    if n_zero == 0:
        if n_neg == 0:
            definiteness = "positive_definite"
        elif n_pos == 0:
            definiteness = "negative_definite"
        else:
            definiteness = "indefinite"
    elif n_neg == 0:
        definiteness = "positive_semidefinite"
    elif n_pos == 0:
        definiteness = "negative_semidefinite"
    else:
        definiteness = "indefinite"
    return InertiaResult(
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
        definiteness=definiteness,
    )


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
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
            y_t_b=_format_rational(ytb),
            reason="multiplier vector has a negative entry",
        )

    n_vars = len(constraint_matrix[0])
    yta = [Fraction(0)] * n_vars
    for i, yi in enumerate(y):
        for j in range(n_vars):
            yta[j] += yi * constraint_matrix[i][j]
    ytb = sum((yi * bi for yi, bi in zip(y, b, strict=True)), Fraction(0))
    yta_str = tuple(_format_rational(value) for value in yta)

    if all(value == 0 for value in yta) and ytb < 0:
        return FarkasCertificateResult(
            valid=True,
            y_t_a=yta_str,
            y_t_b=_format_rational(ytb),
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
        y_t_b=_format_rational(ytb),
        reason="; ".join(reasons) if reasons else "unknown",
    )


__all__ = ["check_farkas_certificate", "compute_inertia"]
