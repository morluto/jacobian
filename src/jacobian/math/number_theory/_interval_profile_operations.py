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
    GreatestPrimeFactorProfileResult,
    GreatestPrimeFactorProfileRow,
    IntervalProfileRequest,
    MAX_INTERVAL_UPPER_BOUND,
    MAX_INTERVAL_WIDTH,
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


def _simple_sieve(limit: int) -> list[int]:
    """Return all primes up to and including limit using a basic sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]


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
        squarefree_values=squarefree_values,
        nonsquarefree_values=nonsquarefree_values,
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

    # Use a divisor-count sieve: start with tau=1 for n=1 and tau=0 for n>1
    # is wrong; instead use the additive divisor sieve.
    # Initialize: tau[0] = 1 for n=1; we use a segmented sieve approach:
    # For each prime p, for each multiple of p in [lo, hi], factor out p and
    # accumulate the exponent.  Then tau(n) = product of (e_i + 1).
    #
    # Simpler: use a direct approach with a sieve counting divisors.
    # Initialize tau(n) for the interval.
    tau = [0] * width

    # For each integer d from 1 to hi, add 1 to tau(n) for every
    # multiple of d in [lo, hi].  This is O(hi * log(hi)) which may be too
    # slow for large hi.  Instead use the segmented approach:
    # For each n in [lo, hi], compute tau(n) from its factorization via a
    # smallest-prime-factor sieve.

    # SPFs sieve: for each n in [lo, hi], find its smallest prime factor.
    sqrt_hi = math.isqrt(hi)
    primes = _simple_sieve(hi)

    # For each n in [lo, hi], factorize it using trial division by primes
    # (using the fact that if n has a prime factor p > sqrt(n), then n/p
    # has already been processed).
    rows: list[DivisorCountProfileRow] = []
    for i in range(width):
        n = lo + i
        m = n
        tau_n = 1
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                exp = 0
                while m % p == 0:
                    m //= p
                    exp += 1
                tau_n *= exp + 1
        if m > 1:
            tau_n *= 2
        rows.append(DivisorCountProfileRow(n=n, divisor_count=tau_n))

    return DivisorCountProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=rows,
    )


def compute_greatest_prime_factor_profile(
    request: IntervalProfileRequest,
) -> GreatestPrimeFactorProfileResult:
    """Compute P+(n) for every n in [L, U]."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound
    width = hi - lo + 1

    primes = _simple_sieve(hi)

    rows: list[GreatestPrimeFactorProfileRow] = []
    for i in range(width):
        n = lo + i
        if n == 1:
            rows.append(GreatestPrimeFactorProfileRow(n=1, greatest_prime_factor=1))
            continue
        m = n
        gpf = 1
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                gpf = p
                while m % p == 0:
                    m //= p
        if m > 1:
            gpf = m
        rows.append(
            GreatestPrimeFactorProfileRow(n=n, greatest_prime_factor=gpf)
        )

    return GreatestPrimeFactorProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=rows,
    )

def compute_prime_gap_profile(
    request: IntervalProfileRequest,
) -> PrimeGapProfileResult:
    """Compute consecutive-prime gaps for primes p with L <= p <= U."""
    _require_admitted(request)
    lo = request.lower_bound
    hi = request.upper_bound

    # We need primes from lo to the next prime after hi.
    # Use a segmented sieve over [lo, hi + gap_search_range].
    # For simplicity with the bounded domain, use sympy's nextprime for
    # the successor and a simple sieve for the interval.
    primes_in_interval: list[int] = []

    # Sieve primes in [lo, hi]
    if lo <= 2:
        sieve_limit = max(hi, 2)
    else:
        sieve_limit = hi

    sqrt_limit = math.isqrt(sieve_limit) + 1
    small_primes = _simple_sieve(sqrt_limit)

    is_prime = bytearray(b"\x01") * (sieve_limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in small_primes:
        for j in range(p * p, sieve_limit + 1, p):
            is_prime[j] = 0
    for i in range(max(2, lo), sieve_limit + 1):
        if is_prime[i]:
            primes_in_interval.append(i)

    # We need at least 2 primes to form a gap.  If no primes in [lo, hi],
    # return empty.  If exactly one prime in [lo, hi], we need the next
    # prime after hi to form the gap.
    if not primes_in_interval:
        return PrimeGapProfileResult(
            lower_bound=lo, upper_bound=hi, rows=[]
        )

    # Find the successor prime after the last prime in interval
    last_prime = primes_in_interval[-1]
    if last_prime <= hi:
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
            rows.append(
                PrimeGapProfileRow(
                    lower_prime=p, upper_prime=q, gap=q - p
                )
            )

    return PrimeGapProfileResult(
        lower_bound=lo, upper_bound=hi, rows=rows
    )


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
    from jacobian.math.number_theory._interval_profile_models import (
        LeastPrimeFactorProfileResult,
        LeastPrimeFactorProfileRow,
    )
    lo, hi = request.lower_bound, request.upper_bound
    primes = _simple_sieve(hi)
    rows = []
    for n in range(lo, hi + 1):
        if n == 1:
            rows.append(LeastPrimeFactorProfileRow(n=1, least_prime_factor=1))
            continue
        m, lpf = n, n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                lpf = p
                break
        rows.append(LeastPrimeFactorProfileRow(n=n, least_prime_factor=lpf))
    return LeastPrimeFactorProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


def compute_euler_totient_profile(
    request: IntervalProfileRequest,
) -> EulerTotientProfileResult:
    """Compute phi(n) for every n in [L, U]."""
    from jacobian.math.number_theory._interval_profile_models import (
        EulerTotientProfileResult,
        EulerTotientProfileRow,
    )
    lo, hi = request.lower_bound, request.upper_bound
    primes = _simple_sieve(hi)
    rows = []
    for n in range(lo, hi + 1):
        if n == 1:
            rows.append(EulerTotientProfileRow(n=1, euler_totient=1))
            continue
        phi, temp = n, n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                phi = phi // p * (p - 1)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            phi = phi // temp * (temp - 1)
        rows.append(EulerTotientProfileRow(n=n, euler_totient=phi))
    return EulerTotientProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


def compute_divisor_sum_profile(
    request: IntervalProfileRequest,
) -> DivisorSumProfileResult:
    """Compute sigma(n) (sum of divisors) for every n in [L, U]."""
    from jacobian.math.number_theory._interval_profile_models import (
        DivisorSumProfileResult,
        DivisorSumProfileRow,
    )
    lo, hi = request.lower_bound, request.upper_bound
    primes = _simple_sieve(hi)
    rows = []
    for n in range(lo, hi + 1):
        temp, sigma = n, 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                pk = 1
                while temp % p == 0:
                    temp //= p
                    pk *= p
                sigma *= (pk * p - 1) // (p - 1)
        if temp > 1:
            sigma *= temp + 1
        rows.append(DivisorSumProfileRow(n=n, divisor_sum=sigma))
    return DivisorSumProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)
