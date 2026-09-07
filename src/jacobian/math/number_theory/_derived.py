"""Exact number-theory operations with structured, argument-bound results."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._derived_models import (
    BinomialPrimeValuationRequest,
    BinomialPrimeValuationResult,
    FactorialValuationRequest,
    FactorialValuationResult,
    FloorSquareRootRequest,
    FloorSquareRootResult,
    LegendreSymbolRequest,
    LegendreSymbolResult,
    admit_binomial_prime_valuation,
    admit_factorial_valuation,
)
from jacobian.math.number_theory.operations import (
    _binomial_prime_valuation,
    _factorial_valuation,
    floor_square_root,
    legendre_symbol,
)


def compute_floor_square_root(request: FloorSquareRootRequest) -> FloorSquareRootResult:
    return floor_square_root(request.n)


def compute_legendre_symbol(request: LegendreSymbolRequest) -> LegendreSymbolResult:
    return legendre_symbol(request.a, request.prime)


def compute_factorial_valuation(
    request: FactorialValuationRequest,
) -> FactorialValuationResult:
    return _factorial_valuation(
        admit_factorial_valuation(
            request.n,
            request.base,
        )
    )


def compute_binomial_prime_valuation(
    request: BinomialPrimeValuationRequest,
) -> BinomialPrimeValuationResult:
    return _binomial_prime_valuation(
        admit_binomial_prime_valuation(
            request.n,
            request.k,
            request.prime,
        )
    )


DERIVED_NUMBER_THEORY_OPERATIONS = (
    MathTool(
        operation_id="number_theory.binomial_valuation.compute",
        title="Compute one binomial prime valuation",
        description="Return the exact exponent of a prime in C(n, k) using Kummer carries without constructing the binomial coefficient.",
        request_type=BinomialPrimeValuationRequest,
        result_type=BinomialPrimeValuationResult,
        run=compute_binomial_prime_valuation,
        tags=("number-theory", "valuation", "binomial"),
        discovery_terms=("Kummer theorem", "p-adic valuation of n choose k"),
        examples=(
            OperationExample(
                name="valuation_of_8_choose_3_at_2",
                description="Compute the exponent of 2 in C(8, 3).",
                input={"n": "8", "k": "3", "prime": "2"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.floor_square_root",
        title="Compute an integer floor square root",
        description="Return floor(sqrt(n)) exactly for a bounded nonnegative integer.",
        request_type=FloorSquareRootRequest,
        result_type=FloorSquareRootResult,
        run=compute_floor_square_root,
        tags=("number-theory", "square"),
        examples=(
            OperationExample(
                name="floor_sqrt_80",
                description="Compute floor(sqrt(80)).",
                input={"n": 80},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.compute.legendre_symbol",
        title="Compute a Legendre symbol",
        description="Compute (a/p) exactly for a bounded odd prime p.",
        request_type=LegendreSymbolRequest,
        result_type=LegendreSymbolResult,
        run=compute_legendre_symbol,
        tags=("number-theory", "quadratic-residue"),
        examples=(
            OperationExample(
                name="legendre_2_mod_7",
                description="Compute the Legendre symbol (2/7).",
                input={"a": 2, "prime": 7},
            ),
            OperationExample(
                name="legendre_3_mod_11",
                description="Compute the Legendre symbol (3/11); the prime denominator must be odd.",
                input={"a": 3, "prime": 11},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.compute.factorial_valuation",
        title="Compute a factorial valuation",
        description="Compute the largest e such that base**e divides n!.",
        request_type=FactorialValuationRequest,
        result_type=FactorialValuationResult,
        run=compute_factorial_valuation,
        tags=("number-theory", "valuation"),
        examples=(
            OperationExample(
                name="valuation_10_factorial_base_2",
                description="Compute the exponent of 2 in 10!.",
                input={"n": "10", "base": "2"},
            ),
        ),
    ),
)
