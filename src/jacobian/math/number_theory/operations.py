"""Canonical exact number-theory operations."""

from __future__ import annotations

import operator
from typing import SupportsIndex

from jacobian._exact import CanonicalInteger
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory._integer_models import BooleanResult
from jacobian.math.number_theory._prime_models import PrimorialResult
from jacobian.math.number_theory.arithmetic.values import IntegerValue

__all__ = [
    "euler_totient",
    "is_prime",
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
