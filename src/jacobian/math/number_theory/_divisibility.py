"""Divisibility-owned exact number-theory operations."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._divisibility_models import (
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)
from jacobian.math.number_theory._factorization import FACTORIZATION_OPERATIONS
from jacobian.math.number_theory._integer_models import (
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._support import (
    number_theory_operation,
)
from jacobian.math.number_theory.arithmetic.operations import (
    divisor_count,
    divisor_sum,
    extended_gcd,
    prime_valuation,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_extended_gcd(request: IntegerPairRequest) -> ExtendedGcdResult:
    return extended_gcd(request.left, request.right)


def compute_valuation(request: ValuationRequest) -> IntegerValue:
    return prime_valuation(request.value, request.prime)


def compute_divisor_count(request: PositiveIntegerRequest) -> IntegerValue:
    return divisor_count(request.n)


def compute_divisor_sum(request: PositiveIntegerRequest) -> IntegerValue:
    return divisor_sum(request.n)


DIVISIBILITY_OPERATIONS = (
    *FACTORIZATION_OPERATIONS,
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
        discovery_terms=("sum positive divisors",),
        examples=(
            example("divisor_sum_12", "Sum the positive divisors of 12.", {"n": 12}),
        ),
    ),
)
