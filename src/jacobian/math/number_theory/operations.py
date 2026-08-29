"""Canonical exact number-theory operations."""

from __future__ import annotations

import operator
from typing import Literal, SupportsIndex, cast

from jacobian._exact import CanonicalInteger
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._derived_models import (
    MAX_FACTORIAL_ARGUMENT,
    MAX_FACTORIAL_BASE,
    MAX_LEGENDRE_PRIME,
    FactorialValuationResult,
    FloorSquareRootResult,
    LegendreSymbolResult,
)
from jacobian.math.number_theory._integer_models import BooleanResult
from jacobian.math.number_theory._prime_models import PrimorialResult
from jacobian.math.number_theory.arithmetic.values import IntegerValue

__all__ = [
    "euler_totient",
    "factorial_valuation",
    "floor_square_root",
    "is_prime",
    "legendre_symbol",
    "mobius",
    "next_prime",
    "nth_prime",
    "previous_prime",
    "prime_count",
    "primorial",
]


def _integer(value: SupportsIndex | CanonicalInteger | IntegerValue) -> int:
    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    if isinstance(value, str):
        return parse_canonical_integer(value)
    return operator.index(value)


def is_prime(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether an integer is prime."""

    from sympy import isprime

    return BooleanResult(holds=bool(isprime(_integer(value))))


def next_prime(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the least prime strictly greater than an integer."""

    from sympy import nextprime

    return IntegerValue(value=format_canonical_integer(int(nextprime(_integer(value)))))


def previous_prime(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the greatest prime strictly below an integer."""

    from sympy import prevprime

    return IntegerValue(value=format_canonical_integer(int(prevprime(_integer(value)))))


def prime_count(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the number of primes not exceeding a nonnegative integer."""

    from sympy import primepi

    return IntegerValue(value=format_canonical_integer(int(primepi(_integer(value)))))


def nth_prime(index: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the prime at one-based positive index."""

    from sympy import prime

    return IntegerValue(value=format_canonical_integer(int(prime(_integer(index)))))


def primorial(index: SupportsIndex | CanonicalInteger | IntegerValue) -> PrimorialResult:
    """Return the product of the first ``index`` primes."""

    from sympy import primorial as sympy_primorial

    return PrimorialResult(
        value=format_canonical_integer(int(sympy_primorial(_integer(index))))
    )


def euler_totient(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return Euler's totient of a positive integer."""

    from sympy import totient

    return IntegerValue(value=format_canonical_integer(int(totient(_integer(value)))))


def mobius(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the Mobius function of a positive integer."""

    from sympy import mobius as sympy_mobius

    return IntegerValue(
        value=format_canonical_integer(int(sympy_mobius(_integer(value))))
    )


def floor_square_root(value: int) -> FloorSquareRootResult:
    """Return the exact floor of the square root of a nonnegative integer."""

    if value < 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.floor_square_root.nonnegative_required",
            message="floor square root requires a nonnegative integer",
        )
    from sympy import integer_nthroot

    root, _ = integer_nthroot(value, 2)
    return FloorSquareRootResult(root=int(root))


def legendre_symbol(a: int, prime: int) -> LegendreSymbolResult:
    """Return the Legendre symbol ``(a / prime)`` for an odd prime."""

    from sympy import isprime
    from sympy import legendre_symbol as sympy_legendre_symbol

    if not 3 <= prime <= MAX_LEGENDRE_PRIME or not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.legendre_denominator_must_be_prime",
            message="Legendre denominator must be prime",
        )
    return LegendreSymbolResult(
        a=a,
        prime=prime,
        symbol=cast(Literal[-1, 0, 1], int(sympy_legendre_symbol(a, prime))),
    )


def factorial_valuation(n: int, base: int) -> FactorialValuationResult:
    """Return the largest exponent ``e`` for which ``base**e`` divides ``n!``."""

    if not 0 <= n <= MAX_FACTORIAL_ARGUMENT:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.factorial_valuation.argument_bound",
            message=f"n must be between 0 and {MAX_FACTORIAL_ARGUMENT}",
        )
    if not 2 <= base <= MAX_FACTORIAL_BASE:
        raise OperationDomainValidationError(
            location=("base",),
            code="number_theory.factorial_valuation.base_bound",
            message=f"base must be between 2 and {MAX_FACTORIAL_BASE}",
        )
    from sympy.ntheory import multiplicity_in_factorial

    return FactorialValuationResult(
        n=n,
        base=base,
        valuation=int(multiplicity_in_factorial(base, n)),
    )
