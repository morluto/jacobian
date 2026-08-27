"""Exact rational polynomial multiplication kernel using SymPy."""

from __future__ import annotations

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._multiply_models import (
    RationalPolynomialMultiplyRequest,
    _is_multiplicative_identity,
)
from jacobian.math.polynomials.values import RationalPolynomial


def rational_polynomial_multiply(
    request: RationalPolynomialMultiplyRequest,
) -> RationalPolynomial:
    """Multiply two rational polynomials exactly using SymPy.

    The result is the canonical exact product in the same QQ variable ring,
    with zero coefficients removed and terms in canonical order.
    """
    if _is_multiplicative_identity(request.left):
        return request.right
    if _is_multiplicative_identity(request.right):
        return request.left

    left_sym = rational_polynomial_to_sympy(request.left)
    right_sym = rational_polynomial_to_sympy(request.right)
    product_sym = left_sym * right_sym

    return rational_polynomial_from_sympy(
        product_sym,
        request.left.variables,
    )


__all__ = ["rational_polynomial_multiply"]
