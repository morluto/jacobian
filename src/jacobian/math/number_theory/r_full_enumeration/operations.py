"""r-full enumeration kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.r_full_enumeration._models import (
    MAX_R_FULL_SIEVE_BOUND,
    RFullEnumerationResult,
)

__all__ = ["enumerate_r_full"]


def enumerate_r_full(bound: int, minimum_exponent: int) -> RFullEnumerationResult:
    """Return all positive r-full integers at most *bound*.

    An integer n is r-full when every prime divisor occurs to exponent
    at least r. The enumeration generates all products of prime powers
    p^e with e >= r.
    """
    if bound < 0 or bound > MAX_R_FULL_SIEVE_BOUND:
        raise OperationDomainValidationError(
            location=("bound",),
            code="number_theory.r_full_sieve_bound",
            message=(
                "r-full enumeration supports bounds from 0 through "
                f"{MAX_R_FULL_SIEVE_BOUND}"
            ),
        )
    if minimum_exponent < 2:
        raise OperationDomainValidationError(
            location=("minimum_exponent",),
            code="number_theory.r_full_minimum_exponent",
            message="r-full enumeration requires minimum_exponent at least 2",
        )
    if bound < 1:
        return RFullEnumerationResult(
            bound=bound,
            minimum_exponent=minimum_exponent,
            values=(),
            count=0,
        )

    # Sieve primes up to bound
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(bound**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    primes = [p for p in range(2, bound + 1) if sieve[p]]

    values: set[int] = set()

    # Generate all r-full numbers
    # An r-full number is a product of prime powers p^e with e >= r
    # We need to enumerate all such products up to bound
    _enumerate_recursive(primes, minimum_exponent, bound, 1, 0, values)

    result = sorted(v for v in values if 1 <= v <= bound)
    return RFullEnumerationResult(
        bound=bound,
        minimum_exponent=minimum_exponent,
        values=tuple(result),
        count=len(result),
    )


def _enumerate_recursive(
    primes: list[int],
    r: int,
    bound: int,
    current: int,
    prime_idx: int,
    values: set[int],
) -> None:
    """Recursively generate r-full numbers."""
    values.add(current)
    for i in range(prime_idx, len(primes)):
        p = primes[i]
        remaining_bound = bound // current
        if r > remaining_bound.bit_length() - 1:
            break
        pe = p**r
        if pe > remaining_bound:
            break
        # Add this prime with exponent >= r
        multiplier = pe
        while current * multiplier <= bound:
            _enumerate_recursive(primes, r, bound, current * multiplier, i + 1, values)
            multiplier *= p
