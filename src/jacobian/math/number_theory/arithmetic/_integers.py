"""Integer-owned exact arithmetic operations.

The arithmetic domain owns only integer absolute value, sign, decimal digit
sum/count, base expansion, and integer nth root.  Number-theory integer
operations (gcd, lcm, divisors, primes, predicates, modular arithmetic) are
owned by the number-theory domain (p3) and are NOT included here.
"""

from jacobian.catalog._examples import example
from jacobian.math.number_theory.arithmetic._models import (
    IntegerNthRootRequest,
    IntegerNthRootResult,
)
from jacobian.math.number_theory.arithmetic._operations import (
    decimal_digit_count,
    nth_root,
)
from jacobian.math.number_theory.arithmetic._support import arithmetic_operation
from jacobian.math.number_theory.arithmetic.values import IntegerValue

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
