"""Exact number-theory capabilities with structured, argument-bound results."""

from jacobian.contracts.number_theory import (
    FactorialValuationRequest,
    FactorialValuationResult,
    FloorSquareRootRequest,
    FloorSquareRootResult,
    LegendreSymbolRequest,
    LegendreSymbolResult,
)
from jacobian.domains._examples import example
from jacobian.domains.number_theory._support import number_theory_operation
from jacobian.domains.number_theory.derived_operations import (
    compute_factorial_valuation,
    compute_floor_square_root,
    compute_legendre_symbol,
)

DERIVED_NUMBER_THEORY_CAPABILITIES = (
    number_theory_operation(
        "integer.compute.floor_square_root",
        "Compute an integer floor square root",
        "Return floor(sqrt(n)) exactly for a bounded nonnegative integer.",
        FloorSquareRootRequest,
        FloorSquareRootResult,
        compute_floor_square_root,
        "number-theory",
        "square",
        invocation_examples=(
            example("floor_sqrt_80", "Compute floor(sqrt(80)).", {"n": 80}),
        ),
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
        invocation_examples=(
            example(
                "legendre_2_mod_7",
                "Compute the Legendre symbol (2/7).",
                {"a": 2, "prime": 7},
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
        invocation_examples=(
            example(
                "valuation_10_factorial_base_2",
                "Compute the exponent of 2 in 10!.",
                {"n": 10, "base": 2},
            ),
        ),
    ),
)
