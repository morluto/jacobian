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
    return SOSDecompositionCheckResult(
        is_valid=is_valid,
        polynomial=request.polynomial,
        summands=request.summands,
        computed_sum=computed_sum,
    )


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
    is_psd = all(val >= 0 for val in eigen_matrix.eigenvals())
    return is_symmetric, reconstructs, is_psd


def check_gram_certificate(
    request: GramCertificateRequest,
) -> GramCertificateResult:
    """Check that p = z^T Q z with Q symmetric PSD over QQ."""
    is_symmetric, reconstructs, is_psd = _check_gram_invariants(
        request.polynomial, request.monomial_basis, request.gram_matrix
    )
    is_valid = is_symmetric and reconstructs and is_psd

    return GramCertificateResult(
        is_valid=is_valid,
        is_symmetric=is_symmetric,
        reconstructs_polynomial=reconstructs,
        is_psd=is_psd,
        polynomial=request.polynomial,
        monomial_basis=request.monomial_basis,
        gram_matrix=request.gram_matrix,
    )


__all__ = [
    "check_gram_certificate",
    "check_sos_decomposition",
]
