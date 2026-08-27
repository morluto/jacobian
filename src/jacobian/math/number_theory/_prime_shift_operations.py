"""Exact translated-prime representation profile kernel."""

from __future__ import annotations

import math

from jacobian.math.number_theory._prime_shift_models import (
    PrimeShiftProfileRequest,
    PrimeShiftProfileResult,
    PrimeShiftProfileRow,
)


def _simple_sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]


def compute_prime_shift_profile(
    request: PrimeShiftRequest,
) -> PrimeShiftProfileResult:
    """Compute the translated-prime representation count for every n in [L, U].

    For each n, count representations n = p + 2^k where p is prime and k >= 0.
    This is the de Polignac-style count: for each prime p and each power of 2
    that satisfies 2^k <= n - p, we count one representation.
    """
    lo = request.lower_bound
    hi = request.upper_bound
    primes = _simple_sieve(hi)

    counts = [0] * (hi - lo + 1)

    for p in primes:
        if p > hi:
            break
        k = 0
        power = 1  # 2^0 = 1
        while True:
            n = p + power
            if n > hi:
                break
            if n >= lo:
                counts[n - lo] += 1
            k += 1
            power *= 2

    rows = [
        PrimeShiftProfileRow(n=lo + i, representation_count=counts[i])
        for i in range(hi - lo + 1)
    ]
    return PrimeShiftProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


# Forward reference for type annotation
from jacobian.math.number_theory._prime_shift_models import (
    PrimeShiftProfileRequest as PrimeShiftRequest,
)


__all__ = ["compute_prime_shift_profile"]
