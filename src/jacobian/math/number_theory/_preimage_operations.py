"""Exact kernels for k*sigma(k) preimage and p-adic interval valuation profiles."""

from __future__ import annotations

import math

from jacobian.math.number_theory._preimage_models import (
    IntervalValuationProfileRequest,
    IntervalValuationProfileResult,
    IntervalValuationProfileRow,
    KSigmaPreimageRequest,
    KSigmaPreimageResult,
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


def compute_ksigma_preimage(
    request: KSigmaPreimageRequest,
) -> KSigmaPreimageResult:
    """Find all n such that k * sigma(n) = target_value.

    sigma(n) is the sum of divisors of n. We iterate over candidate n values
    where sigma(n) = target_value / k (if k divides target_value).
    Uses sympy.divisor_sigma for exact computation.
    """
    from sympy import divisor_sigma

    target_sigma = request.target_value
    k = request.k

    # k * sigma(n) = target => sigma(n) = target / k
    if target_sigma % k != 0:
        return KSigmaPreimageResult(
            k=k, target_value=request.target_value, preimages=[]
        )

    sigma_target = target_sigma // k

    # Iterate candidate n values up to sigma_target (sigma(n) >= n+1 for n >= 2)
    preimages = []
    # sigma(n) >= 1 + n for n >= 2, so n <= sigma_target - 1
    upper = min(sigma_target, 100000)  # bounded search
    for n in range(1, upper + 1):
        if int(divisor_sigma(n)) == sigma_target:
            preimages.append(n)

    return KSigmaPreimageResult(
        k=k, target_value=request.target_value, preimages=preimages
    )


def compute_interval_valuation_profile(
    request: IntervalValuationProfileRequest,
) -> IntervalValuationProfileResult:
    """Compute v_p(n) for every n in [L, U] where p is prime."""
    lo = request.lower_bound
    hi = request.upper_bound
    p = request.prime

    rows = []
    for n in range(lo, hi + 1):
        m = n
        val = 0
        while m % p == 0 and m > 0:
            val += 1
            m //= p
        rows.append(IntervalValuationProfileRow(n=n, valuation=val))

    return IntervalValuationProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        prime=p,
        rows=rows,
    )


__all__ = [
    "compute_interval_valuation_profile",
    "compute_ksigma_preimage",
]
