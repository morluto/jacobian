"""Exact number-theory operations with structured, argument-bound results."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._derived_models import (
    FactorialValuationRequest,
    FactorialValuationResult,
    FloorSquareRootRequest,
    FloorSquareRootResult,
    LegendreSymbolRequest,
    LegendreSymbolResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import (
    factorial_valuation,
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
    return factorial_valuation(request.n, request.base)


DERIVED_NUMBER_THEORY_OPERATIONS = (
    number_theory_operation(
        "integer.compute.floor_square_root",
        "Compute an integer floor square root",
        "Return floor(sqrt(n)) exactly for a bounded nonnegative integer.",
        FloorSquareRootRequest,
        FloorSquareRootResult,
        compute_floor_square_root,
        "number-theory",
        "square",
        examples=(example("floor_sqrt_80", "Compute floor(sqrt(80)).", {"n": 80}),),
    ),
    number_theory_operation(
        "number_theory.compute.legendre_symbol",
        "Compute a Legendre symbol",
        "Compute (a/p) exactly for a bounded odd prime p.",
        LegendreSymbolRequest,
        LegendreSymbolResult,
        compute_legendre_symbol,
        "number-theory",
        "quadratic-residue",
        examples=(
            example(
                "legendre_2_mod_7",
                "Compute the Legendre symbol (2/7).",
                {"a": 2, "prime": 7},
            ),
            example(
                "legendre_3_mod_11",
                "Compute the Legendre symbol (3/11); the prime denominator must be odd.",
                {"a": 3, "prime": 11},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.compute.factorial_valuation",
        "Compute a factorial valuation",
        "Compute the largest e such that base**e divides n!.",
        FactorialValuationRequest,
        FactorialValuationResult,
        compute_factorial_valuation,
        "number-theory",
        "valuation",
        examples=(
            example(
                "valuation_10_factorial_base_2",
                "Compute the exponent of 2 in 10!.",
                {"n": 10, "base": 2},
            ),
        ),
    ),
)
