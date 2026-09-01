"""Typed contracts for rational polynomial multiplication."""

from __future__ import annotations

from jacobian._exact import (
    canonical_rational_component_digits,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
)

MAX_MULTIPLY_RESULT_TERMS = MAX_POLYNOMIAL_TERMS
# Keep backend convolution work bounded independently from the exact result
# term limit; sparse supports can produce many products that collect together.
MAX_MULTIPLY_PRODUCT_WORK = 1_000_000


def _is_multiplicative_identity(polynomial: RationalPolynomial) -> bool:
    """Return whether a polynomial is the exact unit of its declared ring."""

    return (
        len(polynomial.polynomial.terms) == 1
        and polynomial.polynomial.terms[0].exponents == (0,) * len(polynomial.variables)
        and polynomial.polynomial.terms[0].coefficient.as_fraction() == 1
    )


def _is_coefficient_one_monomial(polynomial: RationalPolynomial) -> bool:
    """Return whether multiplication only shifts the other polynomial."""

    return (
        len(polynomial.polynomial.terms) == 1
        and polynomial.polynomial.terms[0].coefficient.as_fraction() == 1
    )


def _maximum_polynomial_coefficient_digits(polynomial: RationalPolynomial) -> int:
    """Return the greatest canonical coefficient-component width in a polynomial."""

    return max(
        (
            canonical_rational_component_digits(term.coefficient)
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )


def _maximum_product_coefficient_digits(
    left: RationalPolynomial, right: RationalPolynomial
) -> int:
    """Bound each collected product coefficient before backend execution.

    A coefficient can collect at most ``min(n, m)`` products.  Putting all
    product denominators over one common denominator gives a conservative
    component width of ``k * (left_digits + right_digits)`` plus the decimal
    width needed to add ``k`` numerators.  Multiplication by the exact unit is
    an identity, so it preserves the other operand's coefficient widths.
    """

    if _is_coefficient_one_monomial(left):
        return _maximum_polynomial_coefficient_digits(right)
    if _is_coefficient_one_monomial(right):
        return _maximum_polynomial_coefficient_digits(left)

    product_count = min(
        len(left.polynomial.terms),
        len(right.polynomial.terms),
    )
    if product_count == 0:
        return 1
    left_digits = _maximum_polynomial_coefficient_digits(left)
    right_digits = _maximum_polynomial_coefficient_digits(right)
    return product_count * (left_digits + right_digits) + len(str(product_count))


class RationalPolynomialMultiplyRequest(StrictModel):
    """Two rational polynomials in the same variable ring for exact multiplication."""

    left: RationalPolynomial
    right: RationalPolynomial


__all__ = [
    "MAX_MULTIPLY_PRODUCT_WORK",
    "MAX_MULTIPLY_RESULT_TERMS",
    "RationalPolynomialMultiplyRequest",
]
