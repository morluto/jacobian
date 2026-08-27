"""Typed contracts for rational polynomial multiplication."""

from __future__ import annotations

import math
from typing import Self

from pydantic import model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    canonical_rational_component_digits,
)
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, strict_json_object_size
from jacobian.math.polynomials._models import _validation_error
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
)

MAX_MULTIPLY_RESULT_TERMS = MAX_POLYNOMIAL_TERMS
# Keep backend convolution work bounded independently from the exact result
# term limit; sparse supports can produce many products that collect together.
MAX_MULTIPLY_PRODUCT_WORK = 1_000_000
MAX_MULTIPLY_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _json_array_size(item_sizes: tuple[int, ...]) -> int:
    """Return the encoded size of a JSON array from encoded item sizes."""

    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


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


def _result_wire_upper_bound(
    variables: tuple[str, ...],
    *,
    term_count: int,
    coefficient_digits: int,
    maximum_exponents: tuple[int, ...],
) -> int:
    """Bound the canonical JSON size of the resulting rational polynomial."""

    # A signed decimal component is at most one sign, ``coefficient_digits``
    # digits, and two JSON quotes.  Every canonical rational is an object with
    # both components, including integer-valued coefficients.
    rational_component_size = coefficient_digits + 3
    coefficient_size = strict_json_object_size(
        (
            ("den", rational_component_size),
            ("num", rational_component_size),
        )
    )
    exponent_size = _json_array_size(
        tuple(len(str(exponent)) for exponent in maximum_exponents)
    )
    term_size = strict_json_object_size(
        (
            ("coefficient", coefficient_size),
            ("exponents", exponent_size),
        )
    )
    terms_size = _json_array_size((term_size,) * term_count)
    polynomial_size = strict_json_object_size((("terms", terms_size),))
    variables_size = _json_array_size(
        tuple(len(variable.encode("utf-8")) + 2 for variable in variables)
    )
    return strict_json_object_size(
        (
            ("domain", 4),  # JSON string ``"QQ"``.
            ("polynomial", polynomial_size),
            ("variables", variables_size),
        )
    )


class RationalPolynomialMultiplyRequest(StrictModel):
    """Two rational polynomials in the same variable ring for exact multiplication."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_rings_and_budget(self) -> Self:
        if self.left.variables != self.right.variables:
            raise _validation_error("polynomials must use the same ordered variables")
        product_term_work = len(self.left.polynomial.terms) * len(
            self.right.polynomial.terms
        )
        if product_term_work > MAX_MULTIPLY_PRODUCT_WORK:
            raise _validation_error(
                "the polynomial product exceeds the bounded convolution work limit"
            )
        coefficient_digits = _maximum_product_coefficient_digits(self.left, self.right)
        if coefficient_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _validation_error(
                "the polynomial product may exceed the canonical coefficient digit limit"
            )
        maximum_exponents = tuple(
            max(
                (term.exponents[index] for term in self.left.polynomial.terms),
                default=0,
            )
            + max(
                (term.exponents[index] for term in self.right.polynomial.terms),
                default=0,
            )
            for index in range(len(self.left.variables))
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
                self.left.variables,
                term_count=result_term_bound,
                coefficient_digits=coefficient_digits,
                maximum_exponents=maximum_exponents,
            )
            > MAX_MULTIPLY_RESULT_BYTES
        ):
            raise _validation_error(
                "the polynomial product may exceed the canonical serialized result size"
            )
        return self


__all__ = [
    "MAX_MULTIPLY_PRODUCT_WORK",
    "MAX_MULTIPLY_RESULT_BYTES",
    "MAX_MULTIPLY_RESULT_TERMS",
    "RationalPolynomialMultiplyRequest",
]
