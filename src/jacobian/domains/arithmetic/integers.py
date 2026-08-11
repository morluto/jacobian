"""Integer-owned exact arithmetic capabilities.

The arithmetic domain owns only integer absolute value, sign, decimal digit
sum/count, base expansion, and integer nth root.  Number-theory integer
operations (gcd, lcm, divisors, primes, predicates, modular arithmetic) are
owned by the number-theory domain (p3) and are NOT included here.
"""

from jacobian.contracts.arithmetic import (
    IntegerBaseDigitsRequest,
    IntegerBaseDigitsResult,
    IntegerNthRootRequest,
    IntegerNthRootResult,
    IntegerSignResult,
    IntegerValueRequest,
    IntegerValueResult,
)
from jacobian.domains._examples import example
from jacobian.domains.arithmetic._support import arithmetic_operation
from jacobian.domains.arithmetic.operations import (
    absolute_value,
    base_digits,
    decimal_digit_count,
    decimal_digit_sum,
    nth_root,
    sign,
)

INTEGER_CAPABILITIES = (
    arithmetic_operation(
        "integer.compute.absolute_value",
        "Compute integer absolute value",
        "Compute the exact absolute value of one integer.",
        IntegerValueRequest,
        IntegerValueResult,
        absolute_value,
        "integer",
        "exact",
        invocation_examples=(
            example(
                "absolute_value_negative_42",
                "Compute the absolute value of -42.",
                {"value": "-42"},
            ),
        ),
    ),
    arithmetic_operation(
        "integer.compute.sign",
        "Compute integer sign",
        "Compute -1, 0, or 1 according to one integer's sign.",
        IntegerValueRequest,
        IntegerSignResult,
        sign,
        "integer",
        "exact",
        invocation_examples=(
            example("sign_negative_42", "Compute the sign of -42.", {"value": "-42"}),
        ),
    ),
    arithmetic_operation(
        "integer.compute.decimal_digit_sum",
        "Compute decimal digit sum",
        "Sum decimal digits of one integer's absolute value.",
        IntegerValueRequest,
        IntegerValueResult,
        decimal_digit_sum,
        "integer",
        "representation",
        invocation_examples=(
            example(
                "decimal_digit_sum_12345",
                "Sum the decimal digits of 12345.",
                {"value": "12345"},
            ),
        ),
    ),
    arithmetic_operation(
        "integer.compute.decimal_digit_count",
        "Count decimal digits",
        "Count decimal digits in one integer's absolute value.",
        IntegerValueRequest,
        IntegerValueResult,
        decimal_digit_count,
        "integer",
        "representation",
        invocation_examples=(
            example(
                "decimal_digit_count_12345",
                "Count the decimal digits of 12345.",
                {"value": "12345"},
            ),
        ),
    ),
    arithmetic_operation(
        "integer.transform.base_digits",
        "Expand integer in a base",
        "Return positional digits of one integer in a base from 2 through 10,000.",
        IntegerBaseDigitsRequest,
        IntegerBaseDigitsResult,
        base_digits,
        "integer",
        "representation",
        invocation_examples=(
            example(
                "negative_binary",
                "Expand negative ten in base two.",
                {"value": "-10", "base": 2},
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
        invocation_examples=(
            example(
                "non_exact_cube_root",
                "Floor cube root of 65.",
                {"value": "65", "degree": 3},
            ),
        ),
    ),
)
