"""Exact contiguous-sum representation profile kernel."""

from __future__ import annotations

from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
    ContiguousSumProfileRow,
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
    """
    lo = request.lower_bound
    hi = request.upper_bound

    # Precompute odd divisors count using a sieve
    counts = [1] * (hi + 1)  # every n has the trivial representation
    counts[0] = 0

    for d in range(2, hi + 1):
        for multiple in range(d, hi + 1, d):
            if d % 2 == 1:
                counts[multiple] += 1

    rows = []
    for n in range(lo, hi + 1):
        rows.append(ContiguousSumProfileRow(n=n, representation_count=counts[n]))

    return ContiguousSumProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


__all__ = ["compute_contiguous_sum_profile"]
