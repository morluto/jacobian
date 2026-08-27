"""Divisibility-owned exact number-theory operations."""

from jacobian.catalog._examples import example
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._divisibility_models import (
    DivisibilityRequest,
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)
from jacobian.math.number_theory._divisibility_operations import (
    compute_aliquot_sum,
    compute_divisor_count,
    compute_divisor_sum,
    compute_extended_gcd,
    compute_gcd,
    compute_lcm,
    compute_valuation,
    decide_abundant,
    decide_coprime,
    decide_deficient,
    decide_divides,
    decide_even,
    decide_odd,
    decide_perfect,
    decide_square,
)
from jacobian.math.number_theory._factorization import FACTORIZATION_OPERATIONS
from jacobian.math.number_theory._integer_models import (
    BooleanResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._support import (
    number_theory_operation,
)

DIVISIBILITY_OPERATIONS = (
    *FACTORIZATION_OPERATIONS,
    number_theory_operation(
        "integer.compute.gcd",
        "Compute integer gcd",
        "Compute the nonnegative greatest common divisor of two integers.",
        IntegerPairRequest,
        IntegerValue,
        compute_gcd,
        "number-theory",
        "divisibility",
        examples=(
            example("gcd_84_30", "Compute gcd(84, 30).", {"left": "84", "right": "30"}),
        ),
    ),
    number_theory_operation(
        "integer.compute.lcm",
        "Compute integer lcm",
        "Compute the nonnegative least common multiple of two integers.",
        IntegerPairRequest,
        IntegerValue,
        compute_lcm,
        "number-theory",
        "divisibility",
        examples=(
            example("lcm_12_18", "Compute lcm(12, 18).", {"left": "12", "right": "18"}),
        ),
    ),
    number_theory_operation(
        "integer.compute.extended_gcd",
        "Compute Bezout coefficients",
        "Compute a gcd and exact Bezout coefficients for two integers.",
        IntegerPairRequest,
        ExtendedGcdResult,
        compute_extended_gcd,
        "number-theory",
        "certificate",
        examples=(
            example(
                "bezout_84_30",
                "Compute Bezout coefficients for 84 and 30.",
                {"left": "84", "right": "30"},
            ),
        ),
    ),
    number_theory_operation(
        "integer.compute.valuation",
        "Compute prime-adic valuation",
        "Compute the exponent of a prime in one nonzero integer.",
        ValuationRequest,
        IntegerValue,
        compute_valuation,
        "number-theory",
        "valuation",
        examples=(
            example(
                "valuation_40_at_2",
                "Compute the 2-adic valuation of 40.",
                {"value": "40", "prime": "2"},
            ),
        ),
    ),
    number_theory_operation(
        "integer.compute.divisor_count",
        "Count positive divisors",
        "Compute the number of positive divisors of one positive integer.",
        PositiveIntegerRequest,
        IntegerValue,
        compute_divisor_count,
        "number-theory",
        "divisibility",
        examples=(
            example(
                "divisor_count_36", "Count the positive divisors of 36.", {"n": 36}
            ),
        ),
    ),
    number_theory_operation(
        "integer.compute.divisor_sum",
        "Sum positive divisors",
        (
            "Compute the sum of every positive divisor of one positive integer, "
            "including the integer itself."
        ),
        PositiveIntegerRequest,
        IntegerValue,
        compute_divisor_sum,
        "number-theory",
        "divisibility",
        examples=(
            example("divisor_sum_12", "Sum the positive divisors of 12.", {"n": 12}),
        ),
    ),
    number_theory_operation(
        "integer.compute.aliquot_sum",
        "Compute aliquot sum",
        "Compute the sum of positive proper divisors of one positive integer.",
        PositiveIntegerRequest,
        IntegerValue,
        compute_aliquot_sum,
        "number-theory",
        "divisibility",
        examples=(
            example("aliquot_sum_12", "Compute the aliquot sum of 12.", {"n": 12}),
        ),
    ),
    number_theory_operation(
        "integer.decide.coprime",
        "Decide coprimality",
        "Decide whether two integers have gcd one.",
        IntegerPairRequest,
        BooleanResult,
        decide_coprime,
        "number-theory",
        "predicate",
        examples=(
            example(
                "coprime_14_25",
                "Check whether 14 and 25 are coprime.",
                {"left": "14", "right": "25"},
            ),
        ),
    ),
    number_theory_operation(
        "integer.decide.divides",
        "Decide divisibility",
        "Decide whether the first nonzero integer divides the second.",
        DivisibilityRequest,
        BooleanResult,
        decide_divides,
        "number-theory",
        "predicate",
        examples=(
            example(
                "divides_6_42",
                "Check whether 6 divides 42.",
                {"divisor": "6", "dividend": "42"},
            ),
        ),
    ),
    number_theory_operation(
        "integer.decide.even",
        "Decide evenness",
        "Decide whether one integer is divisible by two.",
        IntegerValue,
        BooleanResult,
        decide_even,
        "integer",
        "predicate",
        examples=(example("even_42", "Check whether 42 is even.", {"value": "42"}),),
    ),
    number_theory_operation(
        "integer.decide.odd",
        "Decide oddness",
        "Decide whether one integer is not divisible by two.",
        IntegerValue,
        BooleanResult,
        decide_odd,
        "integer",
        "predicate",
        examples=(example("odd_41", "Check whether 41 is odd.", {"value": "41"}),),
    ),
    number_theory_operation(
        "integer.decide.square",
        "Decide perfect square",
        "Decide whether a nonnegative integer is a square.",
        NonnegativeIntegerRequest,
        BooleanResult,
        decide_square,
        "number-theory",
        "predicate",
        examples=(
            example("square_144", "Check whether 144 is a perfect square.", {"n": 144}),
        ),
    ),
    number_theory_operation(
        "integer.decide.perfect",
        "Decide perfect number",
        "Decide whether a positive integer equals its aliquot sum.",
        NonnegativeIntegerRequest,
        BooleanResult,
        decide_perfect,
        "number-theory",
        "predicate",
        examples=(example("perfect_28", "Check whether 28 is perfect.", {"n": 28}),),
    ),
    number_theory_operation(
        "integer.decide.abundant",
        "Decide abundant number",
        "Decide whether a positive integer has aliquot sum greater than itself.",
        NonnegativeIntegerRequest,
        BooleanResult,
        decide_abundant,
        "number-theory",
        "predicate",
        examples=(example("abundant_12", "Check whether 12 is abundant.", {"n": 12}),),
    ),
    number_theory_operation(
        "integer.decide.deficient",
        "Decide deficient number",
        "Decide whether a positive integer has aliquot sum below itself.",
        NonnegativeIntegerRequest,
        BooleanResult,
        decide_deficient,
        "number-theory",
        "predicate",
        examples=(
            example("deficient_10", "Check whether 10 is deficient.", {"n": 10}),
        ),
    ),
)
