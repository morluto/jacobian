"""Integer-owned exact arithmetic operations.

The arithmetic domain owns only integer absolute value, sign, decimal digit
sum/count, base expansion, and integer nth root.  Number-theory integer
operations (gcd, lcm, divisors, primes, predicates, modular arithmetic) are
owned by the number-theory domain (p3) and are NOT included here.
"""

from typing import Literal, cast

from jacobian.catalog._examples import example
from jacobian.math.number_theory.arithmetic import operations as native
from jacobian.math.number_theory.arithmetic._models import (
    IntegerBaseDigitsRequest,
    IntegerBaseDigitsResult,
    IntegerNthRootRequest,
    IntegerNthRootResult,
)
from jacobian.math.number_theory.arithmetic._support import arithmetic_operation
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def decimal_digit_count(request: IntegerValue) -> IntegerValue:
    return native.decimal_digit_count(request)


def base_digits(request: IntegerBaseDigitsRequest) -> IntegerBaseDigitsResult:
    sign, base, digits = native.base_digits(
        IntegerValue(value=request.value), request.base
    )
    sign_value = cast(Literal[-1, 0, 1], sign)
    return IntegerBaseDigitsResult(sign=sign_value, base=base, digits=digits)


def nth_root(request: IntegerNthRootRequest) -> IntegerNthRootResult:
    root, exact = native.nth_root(IntegerValue(value=request.value), request.degree)
    return IntegerNthRootResult(root=root.value, exact=exact)


INTEGER_OPERATIONS = (
    arithmetic_operation(
        "integer.compute.decimal_digit_count",
        "Count decimal digits",
        "Count decimal digits in one integer's absolute value.",
        IntegerValue,
        IntegerValue,
        decimal_digit_count,
        "integer",
        "representation",
        examples=(
            example(
                "decimal_digit_count_12345",
                "Count the decimal digits of 12345.",
                {"value": "12345"},
            ),
        ),
    ),
    arithmetic_operation(
        "integer.compute.nth_root",
        "Compute integer nth root",
        "Compute floor nth root and whether it is exact.",
        IntegerNthRootRequest,
        IntegerNthRootResult,
        nth_root,
        "number-theory",
        "root",
        examples=(
            example(
                "non_exact_cube_root",
                "Floor cube root of 65.",
                {"value": "65", "degree": 3},
            ),
        ),
    ),
)
