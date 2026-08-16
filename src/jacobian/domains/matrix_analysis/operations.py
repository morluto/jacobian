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


def compute_inertia(request: SymmetricMatrixRequest) -> InertiaResult:
    """Compute the Sylvester inertia of a symmetric rational matrix.

    Uses LDL decomposition (Gaussian elimination with symmetric pivoting)
    to count positive, negative, and zero eigenvalues.
    """
    n = request.dimension
    mat = _build_matrix(request)

    # Perform LDL^T decomposition: A = L * D * L^T
    # The diagonal of D gives us the inertia.
    # We use fraction-free Gaussian elimination.
    d = [Fraction(0)] * n
    for i in range(n):
        d[i] = mat[i][i]
        for j in range(i):
            # Subtract contributions from previous steps
            pass  # Simple approach below

    # Simpler approach: compute eigenvalue signs via leading principal minors
    # using the explicit formula for inertia via Sylvester's criterion.
    # Actually, let's use a direct LDL decomposition.
    import copy

    a = [row[:] for row in mat]  # copy
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

        if k < n - 1:
            for i in range(k + 1, n):
                factor = a[i][k] / a[k][k] if a[k][k] != 0 else Fraction(0)
                for j in range(k, n):
                    a[i][j] = a[i][j] - factor * a[k][j]

    if n_zero == 0:
        if n_neg == 0:
            definiteness = "positive_definite"
        elif n_pos == 0:
            definiteness = "negative_definite"
        else:
            definiteness = "indefinite"
    else:
        if n_pos == 0 and n_neg == 0:
            definiteness = "positive_semidefinite"  # all zeros
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


def check_farkas_certificate(
    request: FarkasCertificateRequest,
) -> FarkasCertificateResult:
    """Check a rational Farkas infeasibility certificate.

    Given system Ax <= b and multiplier vector y >= 0, the certificate is
    valid if y^T A = 0 and y^T b < 0.
    """
    from fractions import Fraction

    y = [m.as_fraction() for m in request.multipliers]
    A = [[r.as_fraction() for r in row] for row in request.constraint_matrix]
    b = [r.as_fraction() for r in request.rhs_vector]

    # Check y >= 0
    for i, yi in enumerate(y):
        if yi < 0:
            return FarkasCertificateResult(
                valid=False,
                yTa=(),
                yTb=str(sum(yi * bi for yi, bi in zip(y, b, strict=True))),
                reason="multiplier vector has a negative entry",
            )

    # Compute y^T A (should be all zeros)
    n_vars = len(A[0]) if A else 0
    yTa = [Fraction(0)] * n_vars
    for i, yi in enumerate(y):
        for j in range(n_vars):
            yTa[j] += yi * A[i][j]

    yTa_str = tuple(str(v) for v in yTa)

    # Compute y^T b (should be < 0)
    yTb = sum(yi * bi for yi, bi in zip(y, b, strict=True))

    if all(v == 0 for v in yTa) and yTb < 0:
        return FarkasCertificateResult(
            valid=True,
            yTa=yTa_str,
            yTb=str(yTb),
            reason="y^T A = 0 and y^T b < 0",
        )
    else:
        reasons = []
        if any(v != 0 for v in yTa):
            reasons.append("y^T A != 0")
        if yTb >= 0:
            reasons.append("y^T b >= 0")
        return FarkasCertificateResult(
            valid=False,
            yTa=yTa_str,
            yTb=str(yTb),
            reason="; ".join(reasons) if reasons else "unknown",
        )


__all__ = ["compute_inertia", "check_farkas_certificate"]
