"""Divisibility-owned exact number-theory operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._divisibility_models import (
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)
from jacobian.math.number_theory._factorization import FACTORIZATION_OPERATIONS
from jacobian.math.number_theory._integer_models import (
    PositiveIntegerRequest,
)
from jacobian.math.number_theory.arithmetic.operations import (
    divisor_count,
    divisor_sum,
    extended_gcd,
    integer_gcd,
    prime_valuation,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_extended_gcd(request: IntegerPairRequest) -> ExtendedGcdResult:
    return extended_gcd(request.left, request.right)


def compute_gcd(request: IntegerPairRequest) -> IntegerValue:
    return integer_gcd(request.left, request.right)


def compute_valuation(request: ValuationRequest) -> IntegerValue:
    return prime_valuation(request.value, request.prime)


def compute_divisor_count(request: PositiveIntegerRequest) -> IntegerValue:
    return divisor_count(request.n)


def compute_divisor_sum(request: PositiveIntegerRequest) -> IntegerValue:
    return divisor_sum(request.n)


DIVISIBILITY_OPERATIONS = (
    *FACTORIZATION_OPERATIONS,
    MathTool(
        operation_id="integer.compute.gcd",
        title="Compute integer gcd",
        description="Compute the nonnegative greatest common divisor of two integers.",
        request_type=IntegerPairRequest,
        result_type=IntegerValue,
        run=compute_gcd,
        tags=("number-theory", "divisibility"),
        examples=(
            OperationExample(
                name="gcd_84_30",
                description="Compute gcd(84, 30).",
                input={"left": "84", "right": "30"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.extended_gcd",
        title="Compute Bezout coefficients",
        description="Compute a gcd and exact Bezout coefficients for two integers.",
        request_type=IntegerPairRequest,
        result_type=ExtendedGcdResult,
        run=compute_extended_gcd,
        tags=("number-theory", "certificate"),
        examples=(
            OperationExample(
                name="bezout_84_30",
                description="Compute Bezout coefficients for 84 and 30.",
                input={"left": "84", "right": "30"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.valuation",
        title="Compute prime-adic valuation",
        description="Compute the exponent of a prime in one nonzero integer.",
        request_type=ValuationRequest,
        result_type=IntegerValue,
        run=compute_valuation,
        tags=("number-theory", "valuation"),
        examples=(
            OperationExample(
                name="valuation_40_at_2",
                description="Compute the 2-adic valuation of 40.",
                input={"value": "40", "prime": "2"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.divisor_count",
        title="Count positive divisors",
        description="Compute the number of positive divisors of one positive integer.",
        request_type=PositiveIntegerRequest,
        result_type=IntegerValue,
        run=compute_divisor_count,
        tags=("number-theory", "divisibility"),
        examples=(
            OperationExample(
                name="divisor_count_36",
                description="Count the positive divisors of 36.",
                input={"n": 36},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.divisor_sum",
        title="Sum positive divisors",
        description=(
            "Compute the sum of every positive divisor of one positive integer, "
            "including the integer itself."
        ),
        request_type=PositiveIntegerRequest,
        result_type=IntegerValue,
        run=compute_divisor_sum,
        tags=("number-theory", "divisibility"),
        discovery_terms=("sum positive divisors",),
        examples=(
            OperationExample(
                name="divisor_sum_12",
                description="Sum the positive divisors of 12.",
                input={"n": 12},
            ),
        ),
    ),
)
