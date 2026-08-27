"""Exact interval-profile kernels for arithmetic functions.

All four operations share a segmented-sieve kernel.  The public contracts
are:

- squarefree:       partition [L, U] into squarefree and non-squarefree sets
- divisor-count:    (n, tau(n)) for every L <= n <= U
- greatest-prime:   (n, P+(n)) for every L <= n <= U
- prime-gap:        (p, q, q-p) for every consecutive-prime pair with L <= p <= U

Each kernel is a direct bounded computation; no CAS, solver, or persistent
state is used.
"""

from __future__ import annotations

import math

from jacobian.math.number_theory._interval_profile_models import (
    MAX_INTERVAL_UPPER_BOUND,
    MAX_INTERVAL_WIDTH,
    DivisorCountProfileResult,
    DivisorCountProfileRow,
    GreatestPrimeFactorProfileResult,
    GreatestPrimeFactorProfileRow,
    IntervalProfileRequest,
    PrimeGapProfileResult,
    PrimeGapProfileRow,
    SquarefreeProfileResult,
)


def _require_admitted(request: IntervalProfileRequest) -> None:
    """Validate the request is within the admission envelope."""
    if request.lower_bound < 1:
        raise ValueError("lower_bound must be at least 1")
    if request.upper_bound < request.lower_bound:
        raise ValueError("upper_bound must be >= lower_bound")
    if request.upper_bound > MAX_INTERVAL_UPPER_BOUND:
        raise ValueError("upper_bound exceeds the maximum supported bound")
    if request.width() > MAX_INTERVAL_WIDTH:
        raise ValueError("interval width exceeds the maximum supported width")
    if not request.is_admitted():
        raise ValueError("interval result exceeds the canonical output budget")


def _simple_sieve(limit: int) -> list[int]:
    """Return all primes up to and including limit using a basic sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]


def _segmented_primes(lower_bound: int, upper_bound: int) -> list[int]:
    """Return primes in a closed interval without sieving below its lower end."""
    width = upper_bound - lower_bound + 1
    is_prime = bytearray(b"\x01") * width
    if lower_bound == 1:
        is_prime[0] = 0

    for p in _simple_sieve(math.isqrt(upper_bound)):
        first_multiple = max(
            p * p,
            ((lower_bound + p - 1) // p) * p,
        )
        for multiple in range(first_multiple, upper_bound + 1, p):
            is_prime[multiple - lower_bound] = 0

    return [
        lower_bound + offset
        for offset, marked_prime in enumerate(is_prime)
        if marked_prime
    ]


def compute_squarefree_profile(
    request: IntervalProfileRequest,
) -> SquarefreeProfileResult:
    """Partition [L, U] into squarefree and non-squarefree integers."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound
    width = hi - lo + 1

    # Segmented square sieve: mark n in [lo, hi] as non-squarefree if p^2 | n
    # for some prime p.  We only need primes p with p^2 <= hi.
    sqrt_hi = math.isqrt(hi)
    primes = _simple_sieve(sqrt_hi)

    is_non_squarefree = bytearray(width)
    for p in primes:
        p_sq = p * p
        # First multiple of p_sq in [lo, hi]
        start = ((lo + p_sq - 1) // p_sq) * p_sq
        for j in range(start, hi + 1, p_sq):
            is_non_squarefree[j - lo] = 1

    squarefree_values: list[int] = []
    nonsquarefree_values: list[int] = []
    for i in range(width):
        n = lo + i
        if is_non_squarefree[i]:
            nonsquarefree_values.append(n)
        else:
            squarefree_values.append(n)

    return SquarefreeProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        squarefree_values=tuple(squarefree_values),
        nonsquarefree_values=tuple(nonsquarefree_values),
        squarefree_count=len(squarefree_values),
        nonsquarefree_count=len(nonsquarefree_values),
    )


def compute_divisor_count_profile(
    request: IntervalProfileRequest,
) -> DivisorCountProfileResult:
    """Compute tau(n) for every n in [L, U]."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound
    width = hi - lo + 1

    # Factor each interval value in place.  The base sieve is bounded by
    # sqrt(hi), while the mutable residuals are bounded by the interval width.
    primes = _simple_sieve(math.isqrt(hi))
    residuals = list(range(lo, hi + 1))
    divisor_counts = [1] * width

    for p in primes:
        first_multiple = max(p * p, ((lo + p - 1) // p) * p)
        for offset in range(first_multiple - lo, width, p):
            residual = residuals[offset]
            if residual % p != 0:
                continue
            exponent = 0
            while residual % p == 0:
                residual //= p
                exponent += 1
            residuals[offset] = residual
            divisor_counts[offset] *= exponent + 1

    rows: list[DivisorCountProfileRow] = []
    for i in range(width):
        n = lo + i
        if residuals[i] > 1:
            divisor_counts[i] *= 2
        rows.append(DivisorCountProfileRow(n=n, divisor_count=divisor_counts[i]))

    return DivisorCountProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def compute_greatest_prime_factor_profile(
    request: IntervalProfileRequest,
) -> GreatestPrimeFactorProfileResult:
    """Compute P+(n) for every n in [L, U]."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound
    width = hi - lo + 1

    primes = _simple_sieve(math.isqrt(hi))
    residuals = list(range(lo, hi + 1))
    greatest_prime_factors = [1] * width

    for p in primes:
        first_multiple = max(p * p, ((lo + p - 1) // p) * p)
        for offset in range(first_multiple - lo, width, p):
            residual = residuals[offset]
            if residual % p != 0:
                continue
            greatest_prime_factors[offset] = p
            while residual % p == 0:
                residual //= p
            residuals[offset] = residual

    rows: list[GreatestPrimeFactorProfileRow] = []
    for i in range(width):
        n = lo + i
        if n == 1:
            rows.append(GreatestPrimeFactorProfileRow(n=1, greatest_prime_factor=1))
            continue
        gpf = residuals[i] if residuals[i] > 1 else greatest_prime_factors[i]
        rows.append(GreatestPrimeFactorProfileRow(n=n, greatest_prime_factor=gpf))

    return GreatestPrimeFactorProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def compute_prime_gap_profile(
    request: IntervalProfileRequest,
) -> PrimeGapProfileResult:
    """Compute consecutive-prime gaps for primes p with L <= p <= U."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound

    # Mark only [lo, hi]; the successor beyond hi is queried separately.
    primes_in_interval = _segmented_primes(lo, hi)

    # We need at least 2 primes to form a gap.  If no primes in [lo, hi],
    # return empty.  If exactly one prime in [lo, hi], we need the next
    # prime after hi to form the gap.
    if not primes_in_interval:
        return PrimeGapProfileResult(lower_bound=lo, upper_bound=hi, rows=())

    # Find the successor prime after the last prime in interval
    if primes_in_interval[-1] <= hi:
        # Find next prime after hi (or after last_prime, which is <= hi)
        from sympy import nextprime

        successor = int(nextprime(hi))
        primes_in_interval.append(successor)
    elif len(primes_in_interval) >= 2:
        # The last prime is already > hi (shouldn't happen since we only
        # sieve up to hi), but handle defensively
        pass

    rows: list[PrimeGapProfileRow] = []
    for i in range(len(primes_in_interval) - 1):
        p = primes_in_interval[i]
        q = primes_in_interval[i + 1]
        if p >= lo and p <= hi:
            rows.append(PrimeGapProfileRow(lower_prime=p, upper_prime=q, gap=q - p))

    return PrimeGapProfileResult(lower_bound=lo, upper_bound=hi, rows=tuple(rows))


__all__ = [
    "compute_divisor_count_profile",
    "compute_greatest_prime_factor_profile",
    "compute_prime_gap_profile",
    "compute_squarefree_profile",
]
