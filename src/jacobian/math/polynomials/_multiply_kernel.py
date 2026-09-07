"""Exact rational polynomial multiplication through sparse and dense backends."""

from __future__ import annotations

import math

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import _validation_error
from jacobian.math.polynomials._multiply_models import (
    MAX_MULTIPLY_PRODUCT_WORK,
    MAX_MULTIPLY_RESULT_TERMS,
    _is_multiplicative_identity,
    _maximum_product_coefficient_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _dense_height(polynomial: RationalPolynomial) -> tuple[int, int]:
    denominator = 1
    for term in polynomial.polynomial.terms:
        factor = term.coefficient.den // math.gcd(denominator, term.coefficient.den)
        if (
            denominator.bit_length() + factor.bit_length()
            > 3 * MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise _validation_error(
                "dense denominator clearing exceeds the coefficient digit limit"
            )
        denominator *= factor
    numerator_bits = max(
        (
            abs(term.coefficient.num).bit_length()
            + (denominator // term.coefficient.den).bit_length()
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )
    return numerator_bits, 0 if denominator == 1 else denominator.bit_length()


def _admit(left: RationalPolynomial, right: RationalPolynomial) -> bool:
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    product_term_work = len(left.polynomial.terms) * len(right.polynomial.terms)
    maximum_exponents = tuple(
        max((term.exponents[index] for term in left.polynomial.terms), default=0)
        + max(
            (term.exponents[index] for term in right.polynomial.terms),
            default=0,
        )
        for index in range(len(left.variables))
    )
    support_term_bound = math.prod(exponent + 1 for exponent in maximum_exponents)
    result_term_bound = min(product_term_work, support_term_bound)
    if result_term_bound > MAX_MULTIPLY_RESULT_TERMS:
        raise _validation_error(
            "the polynomial product may exceed the canonical term limit"
        )
    if any(exponent > MAX_POLYNOMIAL_EXPONENT for exponent in maximum_exponents):
        raise _validation_error(
            "the polynomial product may exceed the canonical exponent limit"
        )
    dense = (
        product_term_work > MAX_MULTIPLY_PRODUCT_WORK
        and len(left.variables) == 1
        and maximum_exponents[0] + 1 <= MAX_MULTIPLY_RESULT_TERMS
    )
    if dense:
        left_bits, left_den = _dense_height(left)
        right_bits, right_den = _dense_height(right)
        bits = max(
            left_bits
            + right_bits
            + min(len(left.polynomial.terms), len(right.polynomial.terms)).bit_length(),
            left_den + right_den,
        )
        # A dense output span <=4096 bounds each input allocation. Even a
        # classical integer convolution needs at most 4096^2 products;
        # FLINT may privately select faster multiplication. Charge bit work
        # as well as length, including denominator clearing and extraction.
        work = (maximum_exponents[0] + 1) ** 2 * max(1, bits) ** 2
        if work > 1_000_000_000 or bits // 3 + 1 > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _validation_error(
                "dense polynomial product exceeds the bounded coefficient bit-work limit"
            )
    else:
        if product_term_work > MAX_MULTIPLY_PRODUCT_WORK:
            raise _validation_error(
                "the polynomial product exceeds the bounded convolution work limit"
            )
        if (
            _maximum_product_coefficient_digits(left, right)
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise _validation_error(
                "the polynomial product may exceed the canonical coefficient digit limit"
            )
    return dense


def _run_admission(left: RationalPolynomial, right: RationalPolynomial) -> bool:
    try:
        return _admit(left, right)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc


def rational_polynomial_multiply(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    """Multiply two rational polynomials exactly in a bounded backend regime.

    The result is the canonical exact product in the same QQ variable ring,
    with zero coefficients removed and terms in canonical order.
    """
    dense = _run_admission(left, right)
    if _is_multiplicative_identity(left):
        return right
    if _is_multiplicative_identity(right):
        return left

    if dense:
        from flint import fmpq, fmpq_poly

        def convert(polynomial: RationalPolynomial) -> fmpq_poly:
            degree = max(
                (t.exponents[0] for t in polynomial.polynomial.terms), default=0
            )
            coefficients = [fmpq(0)] * (degree + 1)
            for term in polynomial.polynomial.terms:
                coefficients[term.exponents[0]] = fmpq(
                    *term.coefficient.as_integer_ratio()
                )
            return fmpq_poly(coefficients)

        product = convert(left) * convert(right)
        return RationalPolynomial(
            variables=left.variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        exponents=(index,),
                        coefficient=CanonicalRational.from_integer_ratio(
                            int(product[index].numerator),
                            int(product[index].denominator),
                        ),
                    )
                    for index in range(product.degree(), -1, -1)
                    if product[index]
                )
            ),
        )

    left_sym = rational_polynomial_to_sympy(left)
    right_sym = rational_polynomial_to_sympy(right)
    product_sym = left_sym * right_sym

    return rational_polynomial_from_sympy(
        product_sym,
        left.variables,
    )


__all__ = ["rational_polynomial_multiply"]
