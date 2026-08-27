"""Exact contiguous-sum representation profile kernel."""

from __future__ import annotations

from math import isqrt, prod
from time import monotonic

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory._contiguous_sum_models import (
    MAX_FACTORING_WORK_SECONDS,
    MAX_SEGMENTED_SIEVE_UPPER,
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
    ContiguousSumProfileRow,
)
from jacobian.math.number_theory._factorization_kernels import (
    _bounded_direct_factorization,
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


def _factored_odd_divisor_count(
    value: int, *, timeout_seconds: float = MAX_FACTORING_WORK_SECONDS
) -> int | None:
    """Count odd divisors through the bounded factorization worker."""

    factors = _bounded_direct_factorization(value, timeout_seconds=timeout_seconds)
    if factors is None:
        return None
    return prod(
        factor.power + 1
        for factor in factors
        if parse_canonical_integer(factor.prime) % 2
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
    lo = parse_canonical_integer(request.lower_bound)
    hi = parse_canonical_integer(request.upper_bound)

    if hi <= MAX_SEGMENTED_SIEVE_UPPER:
        counts = _segmented_odd_divisor_counts(lo, hi)
    else:
        counts = []
        factorization_deadline = monotonic() + MAX_FACTORING_WORK_SECONDS
        for n in range(lo, hi + 1):
            remaining = factorization_deadline - monotonic()
            if remaining <= 0:
                count = None
            else:
                count = _factored_odd_divisor_count(n, timeout_seconds=remaining)
            if count is None:
                return ContiguousSumProfileResult._unknown(
                    lower_bound=request.lower_bound,
                    upper_bound=request.upper_bound,
                    detail=(
                        "the bounded factorization worker did not establish "
                        "the complete profile"
                    ),
                )
            counts.append(count)

    rows = []
    for n, count in zip(range(lo, hi + 1), counts, strict=True):
        rows.append(
            ContiguousSumProfileRow(
                n=format_canonical_integer(n), representation_count=count
            )
        )

    return ContiguousSumProfileResult(
        lower_bound=request.lower_bound,
        upper_bound=request.upper_bound,
        rows=tuple(rows),
    )


__all__ = ["compute_contiguous_sum_profile"]
