"""Exact prime-coverage profile kernel."""

from __future__ import annotations

import math

from jacobian.math.number_theory._prime_coverage_models import (
    PrimeCoverageProfileRequest,
    PrimeCoverageProfileResult,
    PrimeCoverageProfileRow,
)


def _simple_sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]


def _segmented_omega(lower_bound: int, upper_bound: int) -> list[int]:
    """Return omega for one interval using a square-root base sieve."""

    width = upper_bound - lower_bound + 1
    residuals = list(range(lower_bound, upper_bound + 1))
    counts = bytearray(width)
    for prime in _simple_sieve(math.isqrt(upper_bound)):
        first = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        for multiple in range(first, upper_bound + 1, prime):
            index = multiple - lower_bound
            residual = residuals[index]
            if residual % prime:
                continue
            counts[index] += 1
            while residual % prime == 0:
                residual //= prime
            residuals[index] = residual
    for index, residual in enumerate(residuals):
        if residual > 1:
            counts[index] += 1
    return list(counts)


def compute_prime_coverage_profile(
    request: PrimeCoverageProfileRequest,
) -> PrimeCoverageProfileResult:
    """Compute omega(n) (distinct prime factor count) for every n in [L, U]."""
    lo = request.lower_bound
    hi = request.upper_bound
    omegas = _segmented_omega(lo, hi)
    rows = []
    for n, omega in zip(range(lo, hi + 1), omegas, strict=True):
        rows.append(PrimeCoverageProfileRow(n=n, distinct_prime_count=omega))
    return PrimeCoverageProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


__all__ = ["compute_prime_coverage_profile"]
