"""Exact rational polynomial multiplication kernel using SymPy."""

from __future__ import annotations

import math

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import _validation_error
from jacobian.math.polynomials._multiply_models import (
    MAX_MULTIPLY_PRODUCT_WORK,
    MAX_MULTIPLY_RESULT_BYTES,
    MAX_MULTIPLY_RESULT_TERMS,
    _is_multiplicative_identity,
    _maximum_product_coefficient_digits,
    _result_wire_upper_bound,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
)


def _admit(left: RationalPolynomial, right: RationalPolynomial) -> None:
    if left.variables != right.variables:
        raise _validation_error("polynomials must use the same ordered variables")
    product_term_work = len(left.polynomial.terms) * len(right.polynomial.terms)
    if product_term_work > MAX_MULTIPLY_PRODUCT_WORK:
        raise _validation_error(
            "the polynomial product exceeds the bounded convolution work limit"
        )
    coefficient_digits = _maximum_product_coefficient_digits(left, right)
    if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise _validation_error(
            "the polynomial product may exceed the canonical coefficient digit limit"
        )
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
    if (
        _result_wire_upper_bound(
            left.variables,
            term_count=result_term_bound,
            coefficient_digits=coefficient_digits,
            maximum_exponents=maximum_exponents,
        )
        > MAX_MULTIPLY_RESULT_BYTES
    ):
        raise _validation_error(
            "the polynomial product may exceed the canonical serialized result size"
        )


def _run_admission(left: RationalPolynomial, right: RationalPolynomial) -> None:
    try:
        _admit(left, right)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc


def rational_polynomial_multiply(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    """Multiply two rational polynomials exactly using SymPy.

    The result is the canonical exact product in the same QQ variable ring,
    with zero coefficients removed and terms in canonical order.
    """
    _run_admission(left, right)
    if _is_multiplicative_identity(left):
        return right
    if _is_multiplicative_identity(right):
        return left

    left_sym = rational_polynomial_to_sympy(left)
    right_sym = rational_polynomial_to_sympy(right)
    product_sym = left_sym * right_sym

    return rational_polynomial_from_sympy(
        product_sym,
        left.variables,
    )


__all__ = ["rational_polynomial_multiply"]
