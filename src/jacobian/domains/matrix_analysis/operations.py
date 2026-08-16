"""Domain adapter for matrix analysis operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.matrix_analysis import (
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
        v = entry.value.as_fraction()
        mat[entry.row][entry.col] = v
        if entry.row != entry.col:
            mat[entry.col][entry.row] = v
    return mat


def compute_inertia(request: SymmetricMatrixRequest) -> InertiaResult:  # noqa: C901
    """Compute the Sylvester inertia of a symmetric rational matrix.

    Uses LDL decomposition (Gaussian elimination with symmetric pivoting)
    to count positive, negative, and zero eigenvalues.
    """
    n = request.dimension
    mat = _build_matrix(request)

    a = [row[:] for row in mat]
    n_pos = 0
    n_neg = 0
    n_zero = 0

    for k in range(n):
        pivot = a[k][k]
        if pivot > 0:
            n_pos += 1
        elif pivot < 0:
            n_neg += 1
        else:
            n_zero += 1

        if k < n - 1 and a[k][k] != 0:
            for i in range(k + 1, n):
                factor = a[i][k] / a[k][k]
                for _ in range(k, n):
                    a[i][_] = a[i][_] - factor * a[k][_]

    if n_zero == 0:
        if n_neg == 0:
            definiteness = "positive_definite"
        elif n_pos == 0:
            definiteness = "negative_definite"
        else:
            definiteness = "indefinite"
    elif (n_pos == 0 and n_neg == 0) or n_neg == 0:
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


def check_farkas_certificate(
    request: FarkasCertificateRequest,
) -> FarkasCertificateResult:
    """Check a rational Farkas infeasibility certificate.

    Given system Ax <= b and multiplier vector y >= 0, the certificate is
    valid if y^T A = 0 and y^T b < 0.
    """
    y = [m.as_fraction() for m in request.multipliers]
    constraint_matrix = [
        [r.as_fraction() for r in row] for row in request.constraint_matrix
    ]
    b = [r.as_fraction() for r in request.rhs_vector]

    for yi in y:
        if yi < 0:
            ytb = sum(yi * bi for yi, bi in zip(y, b, strict=True))
            return FarkasCertificateResult(
                valid=False,
                y_t_a=(),
                y_t_b=str(ytb),
                reason="multiplier vector has a negative entry",
            )

    n_vars = len(constraint_matrix[0]) if constraint_matrix else 0
    yta = [Fraction(0)] * n_vars
    for i, yi in enumerate(y):
        for j in range(n_vars):
            yta[j] += yi * constraint_matrix[i][j]

    ytb = sum(yi * bi for yi, bi in zip(y, b, strict=True))

    yta_str = tuple(str(v) for v in yta)

    if all(v == 0 for v in yta) and ytb < 0:
        return FarkasCertificateResult(
            valid=True,
            y_t_a=yta_str,
            y_t_b=str(ytb),
            reason="y^T A = 0 and y^T b < 0",
        )
    else:
        reasons = []
        if any(v != 0 for v in yta):
            reasons.append("y^T A != 0")
        if ytb >= 0:
            reasons.append("y^T b >= 0")
        return FarkasCertificateResult(
            valid=False,
            y_t_a=yta_str,
            y_t_b=str(ytb),
            reason="; ".join(reasons) if reasons else "unknown",
        )


__all__ = ["check_farkas_certificate", "compute_inertia"]
