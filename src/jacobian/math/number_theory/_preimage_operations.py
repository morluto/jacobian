"""Exact kernels for divisor-sum-product fibers and p-adic profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._preimage_models import (
    MAX_INTERVAL_ENDPOINT_DIGITS,
    MAX_INTERVAL_PROFILE_ROWS,
    DivisorSumProductPreimageRequest,
    DivisorSumProductPreimageResult,
    PAdicIntervalProfileRequest,
    PAdicIntervalProfileResult,
    PAdicIntervalProfileRow,
)


@dataclass(frozen=True, slots=True)
class _PAdicIntervalProfilePlan:
    """One semantically admitted profile request for the kernel."""

    start: int
    length: int
    prime: int
    endpoint: int
    powers: tuple[int, ...]


def _admit_p_adic_interval_profile(
    request: PAdicIntervalProfileRequest,
) -> _PAdicIntervalProfilePlan:
    """Admit one parsed request and retain the powers needed by the kernel."""

    start = parse_canonical_integer(request.start)
    length = parse_canonical_integer(request.length)
    prime = parse_canonical_integer(request.prime)
    if start < 0:
        raise OperationDomainValidationError(
            location=("start",),
            code="number_theory.p_adic_interval_start_must_be_nonnegative",
            message="start must be nonnegative",
        )
    if length < 1:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_length_must_be_positive",
            message="length must be positive",
        )
    if prime < 2:
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.p_adic_interval_prime_must_be_at_least_two",
            message="prime must be at least two",
        )

    from sympy import isprime

    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.p_adic_interval_prime_must_be_prime",
            message="prime must be prime",
        )

    endpoint = start + length
    if len(str(endpoint)) > MAX_INTERVAL_ENDPOINT_DIGITS:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_endpoint_digits",
            message=(
                "interval endpoint must have at most "
                f"{MAX_INTERVAL_ENDPOINT_DIGITS} digits"
            ),
        )

    powers: list[int] = []
    power = 1
    while power <= endpoint:
        powers.append(power)
        power *= prime
    if len(powers) > MAX_INTERVAL_PROFILE_ROWS:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_profile_row_bound",
            message=f"profile needs at most {MAX_INTERVAL_PROFILE_ROWS} rows",
        )
    return _PAdicIntervalProfilePlan(
        start=start,
        length=length,
        prime=prime,
        endpoint=endpoint,
        powers=tuple(powers),
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
    plan = _admit_p_adic_interval_profile(request)

    rows: list[PAdicIntervalProfileRow] = []
    total_valuation = 0
    maximum_valuation = 0
    for valuation, power in enumerate(plan.powers):
        next_power = (
            plan.powers[valuation + 1]
            if valuation + 1 < len(plan.powers)
            else plan.endpoint + 1
        )
        divisible_at_power = plan.endpoint // power - plan.start // power
        divisible_at_next_power = plan.endpoint // next_power - plan.start // next_power
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
