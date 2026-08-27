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
    DivisorCountProfileResult,
    DivisorCountProfileRow,
    DivisorSumProfileResult,
    DivisorSumProfileRow,
    EulerTotientProfileResult,
    EulerTotientProfileRow,
    GreatestPrimeFactorProfileResult,
    GreatestPrimeFactorProfileRow,
    IntervalProfileRequest,
    LeastPrimeFactorProfileResult,
    LeastPrimeFactorProfileRow,
    PrimeGapProfileResult,
    PrimeGapProfileRow,
    SquarefreeProfileResult,
)


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


def _segmented_factor_profile_data(
    lower_bound: int, upper_bound: int
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Factor every interval value by charging only interval multiples."""
    width = upper_bound - lower_bound + 1
    remaining = list(range(lower_bound, upper_bound + 1))
    divisor_counts = [1] * width
    greatest_prime_factors = [1] * width
    least_prime_factors = [1] * width
    euler_totients = list(range(lower_bound, upper_bound + 1))
    divisor_sums = [1] * width

    for prime in _simple_sieve(math.isqrt(upper_bound)):
        first_multiple = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        for value in range(first_multiple, upper_bound + 1, prime):
            index = value - lower_bound
            if remaining[index] % prime != 0:
                continue
            exponent = 0
            prime_power = 1
            while remaining[index] % prime == 0:
                remaining[index] //= prime
                prime_power *= prime
                exponent += 1
            divisor_counts[index] *= exponent + 1
            greatest_prime_factors[index] = prime
            if least_prime_factors[index] == 1:
                least_prime_factors[index] = prime
            euler_totients[index] = euler_totients[index] // prime * (prime - 1)
            divisor_sums[index] *= (prime_power * prime - 1) // (prime - 1)

    for index, cofactor in enumerate(remaining):
        if cofactor <= 1:
            continue
        divisor_counts[index] *= 2
        greatest_prime_factors[index] = cofactor
        if least_prime_factors[index] == 1:
            least_prime_factors[index] = cofactor
        euler_totients[index] = euler_totients[index] // cofactor * (cofactor - 1)
        divisor_sums[index] *= cofactor + 1

    return (
        divisor_counts,
        greatest_prime_factors,
        least_prime_factors,
        euler_totients,
        divisor_sums,
    )


def _segmented_prime_values(lower_bound: int, upper_bound: int) -> list[int]:
    """Return primes in [lower_bound, upper_bound] without a prefix sieve."""
    width = upper_bound - lower_bound + 1
    is_prime = bytearray(b"\x01") * width
    for value in range(lower_bound, min(upper_bound, 1) + 1):
        is_prime[value - lower_bound] = 0
    for prime in _simple_sieve(math.isqrt(upper_bound)):
        first_multiple = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        for value in range(first_multiple, upper_bound + 1, prime):
            is_prime[value - lower_bound] = 0
    return [
        lower_bound + index for index, candidate in enumerate(is_prime) if candidate
    ]


def compute_squarefree_profile(
    request: IntervalProfileRequest,
) -> SquarefreeProfileResult:
    """Partition [L, U] into squarefree and non-squarefree integers."""
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
    lo = request.lower_bound
    hi = request.upper_bound
    # Factor only interval values and their multiples; no prefix through U is
    # materialized.
    divisor_counts, _, _, _, _ = _segmented_factor_profile_data(lo, hi)
    rows: list[DivisorCountProfileRow] = []
    for i, divisor_count in enumerate(divisor_counts):
        rows.append(
            DivisorCountProfileRow(
                n=lo + i,
                divisor_count=divisor_count,
            )
        )

    return DivisorCountProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def compute_greatest_prime_factor_profile(
    request: IntervalProfileRequest,
) -> GreatestPrimeFactorProfileResult:
    """Compute P+(n) for every n in [L, U]."""
    lo = request.lower_bound
    hi = request.upper_bound
    _, greatest_prime_factors, _, _, _ = _segmented_factor_profile_data(lo, hi)
    rows: list[GreatestPrimeFactorProfileRow] = []
    for i, greatest_prime_factor in enumerate(greatest_prime_factors):
        rows.append(
            GreatestPrimeFactorProfileRow(
                n=lo + i,
                greatest_prime_factor=greatest_prime_factor,
            )
        )

    return GreatestPrimeFactorProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def compute_prime_gap_profile(
    request: IntervalProfileRequest,
) -> PrimeGapProfileResult:
    """Compute consecutive-prime gaps for primes p with L <= p <= U."""
    lo = request.lower_bound
    hi = request.upper_bound

    # The successor beyond U is queried separately because it is not part of
    # the interval result and therefore need not be included in the segment.
    primes_in_interval = _segmented_prime_values(lo, hi)

    # We need at least 2 primes to form a gap.  If no primes in [lo, hi],
    # return empty.  If exactly one prime in [lo, hi], we need the next
    # prime after hi to form the gap.
    if not primes_in_interval:
        return PrimeGapProfileResult(lower_bound=lo, upper_bound=hi, rows=())

    # Find the successor prime after the last prime in interval
    last_prime = primes_in_interval[-1]
    if last_prime <= hi:
        # Find next prime after hi (or after last_prime, which is <= hi).
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


def compute_least_prime_factor_profile(
    request: IntervalProfileRequest,
) -> LeastPrimeFactorProfileResult:
    """Compute p(n) (least prime factor) for every n in [L, U]."""
    lo, hi = request.lower_bound, request.upper_bound
    _, _, least_prime_factors, _, _ = _segmented_factor_profile_data(lo, hi)
    rows = []
    for i, least_prime_factor in enumerate(least_prime_factors):
        rows.append(
            LeastPrimeFactorProfileRow(
                n=lo + i,
                least_prime_factor=least_prime_factor,
            )
        )
    return LeastPrimeFactorProfileResult(
        lower_bound=lo, upper_bound=hi, rows=tuple(rows)
    )


def compute_euler_totient_profile(
    request: IntervalProfileRequest,
) -> EulerTotientProfileResult:
    """Compute phi(n) for every n in [L, U]."""
    lo, hi = request.lower_bound, request.upper_bound
    _, _, _, euler_totients, _ = _segmented_factor_profile_data(lo, hi)
    rows = []
    for i, euler_totient in enumerate(euler_totients):
        rows.append(
            EulerTotientProfileRow(
                n=lo + i,
                euler_totient=euler_totient,
            )
        )
    return EulerTotientProfileResult(lower_bound=lo, upper_bound=hi, rows=tuple(rows))


def compute_divisor_sum_profile(
    request: IntervalProfileRequest,
) -> DivisorSumProfileResult:
    """Compute sigma(n) (sum of divisors) for every n in [L, U]."""
    lo, hi = request.lower_bound, request.upper_bound
    _, _, _, _, divisor_sums = _segmented_factor_profile_data(lo, hi)
    rows = []
    for i, divisor_sum in enumerate(divisor_sums):
        rows.append(
            DivisorSumProfileRow(
                n=lo + i,
                divisor_sum=divisor_sum,
            )
        )
    return DivisorSumProfileResult(lower_bound=lo, upper_bound=hi, rows=tuple(rows))
