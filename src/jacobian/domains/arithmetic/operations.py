"""Exact integer and rational operations owned by the arithmetic domain.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic, order,
rounding, representation, and predicates.  Number-theory operations (gcd,
lcm, divisors, primes, modular arithmetic, integer predicates) are owned by
the number-theory domain.

No handrolled algorithms are used where the Python standard library or
SymPy provides a maintained implementation.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.arithmetic import (
    IntegerBaseDigitsRequest,
    IntegerBaseDigitsResult,
    IntegerNthRootRequest,
    IntegerNthRootResult,
    IntegerSignResult,
    IntegerValueRequest,
    IntegerValueResult,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.rationals import (
    RationalComparisonResult,
    RationalContinuedFractionResult,
    RationalIntegerResult,
    RationalPairRequest,
    RationalValueRequest,
    RationalValueResult,
)
from jacobian.math import arithmetic as native_arithmetic


def _int(value: str) -> int:
    return int(value)


def _canonical(value: int) -> str:
    return str(value)


def to_fraction(num: str, den: str) -> Fraction:
    """Build a reduced ``Fraction`` from canonical integer strings."""
    return Fraction(int(num), int(den))


def absolute_value(request: IntegerValueRequest) -> IntegerValueResult:
    return IntegerValueResult(
        value=_canonical(native_arithmetic.absolute_value(_int(request.value)))
    )


def sign(request: IntegerValueRequest) -> IntegerSignResult:
    value = native_arithmetic.sign(_int(request.value))
    if value < 0:
        sign: Literal[-1, 0, 1] = -1
    elif value > 0:
        sign = 1
    else:
        sign = 0
    return IntegerSignResult(sign=sign)


def decimal_digit_sum(request: IntegerValueRequest) -> IntegerValueResult:
    return IntegerValueResult(
        value=_canonical(sum(int(digit) for digit in str(abs(_int(request.value)))))
    )


def decimal_digit_count(request: IntegerValueRequest) -> IntegerValueResult:
    return IntegerValueResult(value=_canonical(len(str(abs(_int(request.value))))))


def base_digits(request: IntegerBaseDigitsRequest) -> IntegerBaseDigitsResult:
    from sympy.ntheory import digits as sympy_digits

    value = _int(request.value)
    signed_base, *expanded = sympy_digits(value, request.base)
    sign: Literal[-1, 0, 1]
    if value == 0:
        sign = 0
    elif signed_base < 0:
        sign = -1
    else:
        sign = 1
    return IntegerBaseDigitsResult(
        sign=sign,
        base=abs(signed_base),
        digits=tuple(str(digit) for digit in expanded),
    )


def nth_root(request: IntegerNthRootRequest) -> IntegerNthRootResult:
    from sympy import integer_nthroot

    if request.value < 0 and request.degree % 2 == 0:
        raise ValueError("even root of a negative integer is not integral-real")
    root, exact = integer_nthroot(abs(request.value), request.degree)
    return IntegerNthRootResult(
        root=_canonical(-root if request.value < 0 else root),
        exact=exact,
    )


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _wire(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def reciprocal(request: RationalValueRequest) -> RationalValueResult:
    try:
        value = native_arithmetic.reciprocal(_fraction(request.value))
    except ZeroDivisionError as exc:
        raise ValueError(str(exc)) from exc
    return RationalValueResult(value=_wire(value))


def negation(request: RationalValueRequest) -> RationalValueResult:
    return RationalValueResult(value=_wire(-_fraction(request.value)))


def rational_absolute_value(request: RationalValueRequest) -> RationalValueResult:
    return RationalValueResult(value=_wire(abs(_fraction(request.value))))


def sum_rationals(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(
            native_arithmetic.sum_rationals(
                _fraction(request.left), _fraction(request.right)
            )
        )
    )


def difference(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(_fraction(request.left) - _fraction(request.right))
    )


def product(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(_fraction(request.left) * _fraction(request.right))
    )


def quotient(request: RationalPairRequest) -> RationalValueResult:
    try:
        value = native_arithmetic.quotient(
            _fraction(request.left), _fraction(request.right)
        )
    except ZeroDivisionError as exc:
        raise ValueError(str(exc)) from exc
    return RationalValueResult(value=_wire(value))


def minimum(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(min(_fraction(request.left), _fraction(request.right)))
    )


def maximum(request: RationalPairRequest) -> RationalValueResult:
    return RationalValueResult(
        value=_wire(max(_fraction(request.left), _fraction(request.right)))
    )


def floor(request: RationalValueRequest) -> RationalIntegerResult:
    return RationalIntegerResult(value=str(math.floor(_fraction(request.value))))


def ceiling(request: RationalValueRequest) -> RationalIntegerResult:
    return RationalIntegerResult(value=str(math.ceil(_fraction(request.value))))


def continued_fraction(
    request: RationalValueRequest,
) -> RationalContinuedFractionResult:
    from sympy import Rational as SympyRational
    from sympy import continued_fraction as sympy_continued_fraction

    value = _fraction(request.value)
    terms = sympy_continued_fraction(SympyRational(value.numerator, value.denominator))
    return RationalContinuedFractionResult(
        terms=tuple(str(int(term)) for term in terms)
    )


def equal(request: RationalPairRequest) -> RationalComparisonResult:
    return RationalComparisonResult(
        holds=_fraction(request.left) == _fraction(request.right)
    )


def less_than(request: RationalPairRequest) -> RationalComparisonResult:
    return RationalComparisonResult(
        holds=_fraction(request.left) < _fraction(request.right)
    )
