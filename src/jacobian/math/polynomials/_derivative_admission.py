"""Source, growth and dense-backend admission for exact QQ differentiation."""

from __future__ import annotations

import re
from math import gcd

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_COMPONENT_LIMIT = 10**MAX_CANONICAL_RATIONAL_DIGITS
_MAX_DERIVATIVE_BIT_WORK = 4_000_000_000_000


def _invalid() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("polynomial",),
        code="polynomial.derivative_source",
        message="differentiation requires a canonical one-variable QQ polynomial",
    )


def _resource(reason: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("polynomial",),
        code=f"polynomial.derivative_{reason}",
        message="polynomial derivative exceeds its canonical output or bit-work envelope",
    )


def admit_derivative_source(polynomial: RationalPolynomial) -> RationalPolynomial:
    """Admit QQ Poly.diff and remove the mathematically irrelevant constant.

    FLINT's dense QQ backend may clear all denominators. With B the sum of
    distinct nonunit denominator bit lengths, a common denominator fits B bits and a
    cleared derivative numerator fits max numerator bits + B + exponent bits.
    Charge dense slots times squared width, plus source canonicality checks,
    before any backend allocation or large rational arithmetic. This also
    bounds the Python dense rational backend, without assuming FLINT is active.
    """
    if (
        not isinstance(polynomial, RationalPolynomial)
        or getattr(polynomial, "domain", None) != "QQ"
        or type(getattr(polynomial, "variables", None)) is not tuple
        or len(polynomial.variables) != 1
        or type(polynomial.variables[0]) is not str
        or len(polynomial.variables[0]) > 32
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", polynomial.variables[0]) is None
        or type(getattr(polynomial, "polynomial", None)) is not SparseRationalPolynomial
        or type(getattr(polynomial.polynomial, "terms", None)) is not tuple
        or len(polynomial.polynomial.terms) > MAX_POLYNOMIAL_TERMS
    ):
        raise _invalid()
    active: list[RationalPolynomialTerm] = []
    source_work = 0
    denominators: set[int] = set()
    numerator_bits = 1
    degree = 0
    previous = MAX_POLYNOMIAL_EXPONENT + 1
    for term in polynomial.polynomial.terms:
        if (
            type(term) is not RationalPolynomialTerm
            or type(getattr(term, "exponents", None)) is not tuple
            or len(term.exponents) != 1
            or type(term.exponents[0]) is not int
            or not 0 <= term.exponents[0] < previous
            or type(getattr(term, "coefficient", None)) is not CanonicalRational
        ):
            raise _invalid()
        previous = exponent = term.exponents[0]
        coefficient = term.coefficient
        if (
            type(getattr(coefficient, "num", None)) is not int
            or type(getattr(coefficient, "den", None)) is not int
            or coefficient.num == 0
            or coefficient.den <= 0
        ):
            raise _invalid()
        width = max(coefficient.num.bit_length(), coefficient.den.bit_length())
        if width > 4 * MAX_CANONICAL_RATIONAL_DIGITS:
            raise _invalid()
        source_work += width**2
        if exponent:
            active.append(term)
            degree = max(degree, exponent)
            numerator_bits = max(numerator_bits, coefficient.num.bit_length())
            if coefficient.den != 1:
                denominators.add(coefficient.den)
    denominator_bits = sum(denominator.bit_length() for denominator in denominators)
    width = numerator_bits + max(1, denominator_bits) + degree.bit_length()
    # The fixed factor charges conversion, differentiation, normalization and
    # output conversion; the dense degree envelope counts zeros as well.
    if 16 * (source_work + (degree + 1) * width**2) > _MAX_DERIVATIVE_BIT_WORK:
        raise _resource("work_bound")
    for term in polynomial.polynomial.terms:
        coefficient = term.coefficient
        if (
            abs(coefficient.num) >= _COMPONENT_LIMIT
            or coefficient.den >= _COMPONENT_LIMIT
            or gcd(coefficient.num, coefficient.den) != 1
        ):
            raise _invalid()
        exponent = term.exponents[0]
        if exponent:
            multiplier = exponent // gcd(exponent, coefficient.den)
            # Reduced a/b maps to a*(e/g)/(b/g). No sum or power is formed,
            # and comparison precedes multiplication beyond the carrier.
            if abs(coefficient.num) > (_COMPONENT_LIMIT - 1) // multiplier:
                raise _resource("output_bound")
    return RationalPolynomial.model_construct(
        domain="QQ",
        variables=polynomial.variables,
        polynomial=SparseRationalPolynomial.model_construct(terms=tuple(active)),
    )
