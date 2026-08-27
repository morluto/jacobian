"""Exact contiguous-sum representation profile kernel."""

from __future__ import annotations

from math import isqrt, prod

from jacobian.math.number_theory._contiguous_sum_models import (
    MAX_SEGMENTED_SIEVE_UPPER,
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
    ContiguousSumProfileRow,
)


def _odd_primes_up_to(limit: int) -> list[int]:
    """Return odd primes up to ``limit`` with a bounded segmented regime."""

    sieve = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for candidate in range(3, isqrt(limit) + 1, 2):
        if sieve[candidate]:
            for composite in range(candidate * candidate, limit + 1, 2 * candidate):
                sieve[composite] = 0
    return [candidate for candidate in range(3, limit + 1, 2) if sieve[candidate]]


def _segmented_odd_divisor_counts(lower_bound: int, upper_bound: int) -> list[int]:
    """Count odd divisors over a dense interval without prefix allocation."""

    width = upper_bound - lower_bound + 1
    residuals = list(range(lower_bound, upper_bound + 1))
    counts = [1] * width
    for prime in _odd_primes_up_to(isqrt(upper_bound)):
        first_multiple = ((lower_bound + prime - 1) // prime) * prime
        for multiple in range(first_multiple, upper_bound + 1, prime):
            index = multiple - lower_bound
            residual = residuals[index]
            exponent = 0
            while residual % prime == 0:
                residual //= prime
                exponent += 1
            if exponent:
                residuals[index] = residual
                counts[index] *= exponent + 1
    for index, residual in enumerate(residuals):
        if residual > 1 and residual % 2:
            counts[index] *= 2
    return counts


def _factored_odd_divisor_count(value: int) -> int:
    """Count odd divisors of one high-magnitude value with SymPy."""

    from sympy import factorint

    return prod(
        exponent + 1 for prime, exponent in factorint(value).items() if prime % 2
    )


def compute_contiguous_sum_profile(
    request: ContiguousSumProfileRequest,
) -> ContiguousSumProfileResult:
    """For each n in [L, U], count representations as a sum of consecutive positive integers.

    A contiguous-sum representation of n is: n = a + (a+1) + ... + (a+k-1)
    for some a >= 1 and k >= 1 (k=1 gives the trivial representation n=n).

    The number of such representations equals the number of odd divisors of n
    that are greater than 1 (or equivalently, the number of ways to factor
    n as (a+b)*(b-a+1)/2 with appropriate constraints).

    A known result: the number of ways to write n as a sum of consecutive
    positive integers equals the number of odd divisors of n (including 1).
    Dense intervals use a segmented odd-factor sieve, while high-magnitude
    narrow intervals use the maintained SymPy factorization backend.
    """
    lo = request.lower_bound
    hi = request.upper_bound

    if hi <= MAX_SEGMENTED_SIEVE_UPPER:
        counts = _segmented_odd_divisor_counts(lo, hi)
    else:
        counts = [_factored_odd_divisor_count(n) for n in range(lo, hi + 1)]

    rows = []
    for n, count in zip(range(lo, hi + 1), counts, strict=True):
        rows.append(ContiguousSumProfileRow(n=n, representation_count=count))

    return ContiguousSumProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


__all__ = ["compute_contiguous_sum_profile"]
