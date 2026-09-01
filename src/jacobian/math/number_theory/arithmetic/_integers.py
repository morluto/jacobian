"""Integer-owned exact arithmetic operations.

The arithmetic domain owns only integer absolute value, sign, decimal digit
sum/count, base expansion, and integer nth root.  Number-theory integer
operations (gcd, lcm, divisors, primes, predicates, modular arithmetic) are
owned by the number-theory domain (p3) and are NOT included here.
"""

from typing import Literal, cast

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.arithmetic import operations as native
from jacobian.math.number_theory.arithmetic._models import (
    IntegerBaseDigitsRequest,
    IntegerBaseDigitsResult,
    IntegerNthRootRequest,
    IntegerNthRootResult,
)
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
    MathTool(
        operation_id="integer.compute.decimal_digit_count",
        title="Count decimal digits",
        description="Count decimal digits in one integer's absolute value.",
        request_type=IntegerValue,
        result_type=IntegerValue,
        run=decimal_digit_count,
        tags=("integer", "representation"),
        examples=(
            OperationExample(
                name="decimal_digit_count_12345",
                description="Count the decimal digits of 12345.",
                input={"value": "12345"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.nth_root",
        title="Compute integer nth root",
        description="Compute floor nth root and whether it is exact.",
        request_type=IntegerNthRootRequest,
        result_type=IntegerNthRootResult,
        run=nth_root,
        tags=("number-theory", "root"),
        examples=(
            OperationExample(
                name="non_exact_cube_root",
                description="Floor cube root of 65.",
                input={"value": "65", "degree": 3},
            ),
        ),
    ),
)
