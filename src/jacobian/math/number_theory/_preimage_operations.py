"""Exact kernels for divisor-sum-product fibers and p-adic profiles."""

from __future__ import annotations

from math import isqrt

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._preimage_models import (
    DivisorSumProductPreimageRequest,
    DivisorSumProductPreimageResult,
    PAdicIntervalProfileRequest,
    PAdicIntervalProfileResult,
    PAdicIntervalProfileRow,
)


def compute_divisor_sum_product_preimage(
    request: DivisorSumProductPreimageRequest,
) -> DivisorSumProductPreimageResult:
    """Compute every positive n with ``n * sigma(n) == target``."""
    from sympy import divisor_sigma

    target = parse_canonical_integer(request.target)
    source_upper_bound = isqrt(target)
    preimages = tuple(
        n
        for n in range(1, source_upper_bound + 1)
        if n * int(divisor_sigma(n)) == target
    )
    return DivisorSumProductPreimageResult._from_kernel(
        request,
        preimages=preimages,
    )


def compute_p_adic_interval_profile(
    request: PAdicIntervalProfileRequest,
) -> PAdicIntervalProfileResult:
    """Compute the valuation histogram on ``[start + 1, start + length]``."""
    start = parse_canonical_integer(request.start)
    length = parse_canonical_integer(request.length)
    prime = parse_canonical_integer(request.prime)
    endpoint = start + length

    rows: list[PAdicIntervalProfileRow] = []
    total_valuation = 0
    maximum_valuation = 0
    power = 1
    valuation = 0
    while power <= endpoint:
        next_power = power * prime
        divisible_at_power = endpoint // power - start // power
        divisible_at_next_power = (
            endpoint // next_power - start // next_power
            if next_power <= endpoint
            else 0
        )
        count = divisible_at_power - divisible_at_next_power
        if count:
            rows.append(
                PAdicIntervalProfileRow(
                    valuation=valuation,
                    count=str(count),
                )
            )
            total_valuation += valuation * count
            maximum_valuation = valuation
        power = next_power
        valuation += 1

    return PAdicIntervalProfileResult._from_kernel(
        request,
        rows=tuple(rows),
        total_valuation=total_valuation,
        maximum_valuation=maximum_valuation,
    )


__all__ = [
    "compute_divisor_sum_product_preimage",
    "compute_p_adic_interval_profile",
]
