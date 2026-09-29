"""Typed contracts for rational polynomial multiplication."""

from __future__ import annotations

from math import gcd

from jacobian._exact import (
    canonical_rational_component_digits,
)
from jacobian._models import StrictModel
from jacobian.canonical import decimal_digit_width
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


def _coefficient_content(polynomial: RationalPolynomial) -> tuple[int, int]:
    """Return the numerator and denominator content of every coefficient.

    Dividing a common factor out of every coefficient leaves the polynomial
    numerically unchanged, so measuring the reduced coefficients bounds the same
    exact product with a smaller width.
    """

    numerator_content = 0
    denominator_content = 0
    for term in polynomial.polynomial.terms:
        numerator_content = gcd(numerator_content, term.coefficient.num)
        denominator_content = gcd(denominator_content, term.coefficient.den)
    return numerator_content, denominator_content


def _reduced_component_widths(
    polynomial: RationalPolynomial, content: tuple[int, int]
) -> tuple[int, int]:
    """Return the widest numerator and denominator with ``content`` divided out.

    Every coefficient is already reduced, so dividing a common factor out of
    the numerators or the denominators keeps each coefficient canonical.
    """

    numerator_content, denominator_content = content
    numerator_width = 1
    denominator_width = 1
    for term in polynomial.polynomial.terms:
        numerator_width = max(
            numerator_width,
            decimal_digit_width(term.coefficient.num // numerator_content),
        )
        denominator_width = max(
            denominator_width,
            decimal_digit_width(term.coefficient.den // denominator_content),
        )
    return numerator_width, denominator_width


def _maximum_product_coefficient_digits(
    left: RationalPolynomial, right: RationalPolynomial
) -> int:
    """Bound each collected product coefficient before backend execution.

    A coefficient can collect at most ``min(n, m)`` products.  Putting all
    product denominators over one common denominator gives a conservative
    component width of ``k * (left_digits + right_digits)`` plus the decimal
    width needed to add ``k`` numerators.  Multiplication by the exact unit is
    an identity, so it preserves the other operand's coefficient widths.

    Common factors cross-cancel between the two operands before the product is
    formed, so ``1/N`` against ``N`` is exactly ``1`` and ``N`` against ``1`` is
    exactly ``N``.  Measuring the operands with their cross-cancelled content
    removed is what keeps two carrier-valid at-limit scalars from being refused
    on a product that never actually grows.
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
    left_content = _coefficient_content(left)
    right_content = _coefficient_content(right)
    # Each operand's numerator content cancels against the other's denominator
    # content, so only what survives reaches the product. Dividing the surviving
    # content out of both operands measures the same exact product.
    product_numerator = left_content[0] * right_content[0]
    product_denominator = left_content[1] * right_content[1]
    common = gcd(product_numerator, product_denominator)
    product_numerator //= common
    product_denominator //= common
    left_widths = _reduced_component_widths(left, left_content)
    right_widths = _reduced_component_widths(right, right_content)
    numerator_bound = (
        product_count * (left_widths[0] + right_widths[0])
        + len(str(product_count))
        + decimal_digit_width(product_numerator)
    )
    denominator_bound = (
        product_count * (left_widths[1] + right_widths[1])
        + len(str(product_count))
        + decimal_digit_width(product_denominator)
    )
    return max(numerator_bound, denominator_bound)


class RationalPolynomialMultiplyRequest(StrictModel):
    """Two rational polynomials in the same variable ring for exact multiplication."""

    left: RationalPolynomial
    right: RationalPolynomial


__all__ = [
    "MAX_MULTIPLY_PRODUCT_WORK",
    "MAX_MULTIPLY_RESULT_TERMS",
    "RationalPolynomialMultiplyRequest",
]
