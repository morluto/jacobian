"""Derived exact number-theory operation kernels."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.contracts.number_theory import (
    FactorialValuationRequest,
    FactorialValuationResult,
    FloorSquareRootRequest,
    FloorSquareRootResult,
    LegendreSymbolRequest,
    LegendreSymbolResult,
)


def compute_floor_square_root(request: FloorSquareRootRequest) -> FloorSquareRootResult:
    from sympy import integer_nthroot

    root, _ = integer_nthroot(request.n, 2)
    return FloorSquareRootResult(root=int(root))


def compute_legendre_symbol(request: LegendreSymbolRequest) -> LegendreSymbolResult:
    from sympy import isprime, legendre_symbol

    if not isprime(request.prime):
        raise ValueError("Legendre denominator must be prime")
    return LegendreSymbolResult(
        a=request.a,
        prime=request.prime,
        symbol=cast(Literal[-1, 0, 1], int(legendre_symbol(request.a, request.prime))),
    )


def compute_factorial_valuation(
    request: FactorialValuationRequest,
) -> FactorialValuationResult:
    from sympy.ntheory import multiplicity_in_factorial

    return FactorialValuationResult(
        n=request.n,
        base=request.base,
        valuation=int(multiplicity_in_factorial(request.base, request.n)),
    )
