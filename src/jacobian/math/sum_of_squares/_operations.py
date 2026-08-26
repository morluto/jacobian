"""Domain-owned sum-of-squares operations."""

from __future__ import annotations

import sympy

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.sum_of_squares._models import (
    GramCertificateRequest,
    GramCertificateResult,
    SOSDecompositionCheckRequest,
    SOSDecompositionCheckResult,
)


def _check_sos_invariants(
    polynomial: RationalPolynomial,
    summands: tuple[RationalPolynomial, ...],
) -> tuple[bool, RationalPolynomial]:
    """Exact replay of SOS decomposition check."""
    p_sympy = rational_polynomial_to_sympy(polynomial).as_expr()
    sum_expr = sympy.Integer(0)
    for summand in summands:
        q_sympy = rational_polynomial_to_sympy(summand).as_expr()
        sum_expr += q_sympy * q_sympy
    is_valid = sympy.expand(p_sympy - sum_expr) == 0
    if is_valid:
        computed_sum = polynomial
    else:
        variables = polynomial.variables
        computed_poly = sympy.Poly(
            sum_expr, *sympy.symbols(list(variables)), domain=sympy.QQ
        )
        computed_sum = rational_polynomial_from_sympy(computed_poly, variables)
    return is_valid, computed_sum


def check_sos_decomposition(
    request: SOSDecompositionCheckRequest,
) -> SOSDecompositionCheckResult:
    """Check that p = q_1^2 + ... + q_r^2 by exact coefficient identity."""
    is_valid, computed_sum = _check_sos_invariants(request.polynomial, request.summands)
    return SOSDecompositionCheckResult._from_kernel(
        is_valid=is_valid,
        polynomial=request.polynomial,
        summands=request.summands,
        computed_sum=computed_sum,
    )


def verify_sos_decomposition_result(result: SOSDecompositionCheckResult) -> bool:
    """Replay an independently supplied SOS claim within its admitted envelope."""

    is_valid, computed_sum = _check_sos_invariants(result.polynomial, result.summands)
    return result.is_valid == is_valid and result.computed_sum == computed_sum


def _check_gram_invariants(
    polynomial: RationalPolynomial,
    monomial_basis: tuple[RationalPolynomial, ...],
    gram_matrix: tuple[tuple[CanonicalRational, ...], ...],
) -> tuple[bool, bool, bool]:
    """Exact replay of Gram certificate checks."""
    matrix = sympy.Matrix(
        [[sympy.Rational(c.as_fraction()) for c in row] for row in gram_matrix]
    )
    is_symmetric = matrix == matrix.T
    z = sympy.Matrix(
        [[rational_polynomial_to_sympy(m).as_expr() for m in monomial_basis]]
    ).T
    reconstructed = (z.T * matrix * z)[0, 0]
    p_sympy = rational_polynomial_to_sympy(polynomial).as_expr()
    reconstructs = sympy.expand(reconstructed - p_sympy) == 0
    eigen_matrix = matrix if is_symmetric else (matrix + matrix.T) / 2
    is_psd = _exact_psd(eigen_matrix)
    return is_symmetric, reconstructs, is_psd


def _exact_psd(matrix: sympy.Matrix) -> bool:
    """Total exact PSD test for a symmetric rational matrix.

    Symmetric Gaussian elimination (Lagrange): a symmetric matrix over QQ
    is positive semidefinite iff every pivot is nonnegative and a zero
    pivot forces the whole remaining row and column to vanish. Unlike an
    eigenvalue computation this terminates on every input — irreducible
    characteristic polynomials raise no backend exception.
    """
    n = matrix.rows
    work = [[sympy.Rational(matrix[i, j]) for j in range(n)] for i in range(n)]
    for k in range(n):
        pivot = work[k][k]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(work[k][j] != 0 for j in range(k + 1, n)) or any(
                work[j][k] != 0 for j in range(k + 1, n)
            ):
                return False
            continue
        for i in range(k + 1, n):
            factor = work[i][k] / pivot
            for j in range(k + 1, n):
                work[i][j] -= factor * work[k][j]
    return True


def check_gram_certificate(
    request: GramCertificateRequest,
) -> GramCertificateResult:
    """Check that p = z^T Q z with Q symmetric PSD over QQ."""
    is_symmetric, reconstructs, is_psd = _check_gram_invariants(
        request.polynomial, request.monomial_basis, request.gram_matrix.entries
    )
    return GramCertificateResult._from_kernel(
        is_symmetric=is_symmetric,
        reconstructs_polynomial=reconstructs,
        is_psd=is_psd,
        polynomial=request.polynomial,
        monomial_basis=request.monomial_basis,
        gram_matrix=request.gram_matrix,
    )


def verify_gram_certificate_result(result: GramCertificateResult) -> bool:
    """Replay an independently supplied Gram-certificate claim."""

    is_symmetric, reconstructs, is_psd = _check_gram_invariants(
        result.polynomial, result.monomial_basis, result.gram_matrix.entries
    )
    return (
        result.is_symmetric == is_symmetric
        and result.reconstructs_polynomial == reconstructs
        and result.is_psd == is_psd
        and result.is_valid == (is_symmetric and reconstructs and is_psd)
    )


__all__ = [
    "check_gram_certificate",
    "check_sos_decomposition",
    "verify_gram_certificate_result",
    "verify_sos_decomposition_result",
]
