"""Exact translated-prime representation profile kernel."""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._prime_shift_models import (
    PrimeShiftProfileRequest as PrimeShiftRequest,
)
from jacobian.math.number_theory._prime_shift_models import (
    PrimeShiftProfileResult,
    _PrimeShiftProfileExecutionPlan,
    require_prime_shift_profile_admission,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def _simple_sieve(limit: int) -> tuple[int, ...]:
    if limit < 2:
        return ()
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return tuple(i for i in range(2, limit + 1) if is_prime[i])


def _segmented_sieve(
    lower_bound: int, upper_bound: int, base_primes: tuple[int, ...]
) -> bytearray:
    """Return primality flags for one closed interval."""
    flags = bytearray(b"\x01") * (upper_bound - lower_bound + 1)

    for prime in base_primes:
        if prime * prime > upper_bound:
            break
        first = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        if first <= upper_bound:
            flags[first - lower_bound :: prime] = b"\x00" * (
                (upper_bound - first) // prime + 1
            )
    return flags


def _compute_prime_shift_profile(
    plan: _PrimeShiftProfileExecutionPlan,
) -> PrimeShiftProfileResult:
    """Run the segmented-sieve kernel for one admitted execution plan."""

    base_limit = plan.base_limit
    counts = [0] * (plan.upper_bound - plan.lower_bound + 1)

    base_primes = _simple_sieve(base_limit)
    for power, candidate_lower, candidate_upper in plan.candidate_intervals:
        flags = _segmented_sieve(candidate_lower, candidate_upper, base_primes)
        for offset, is_prime in enumerate(flags):
            if is_prime:
                counts[candidate_lower + offset + power - plan.lower_bound] += 1

    return PrimeShiftProfileResult._from_kernel(
        lower_bound=plan.lower_bound,
        upper_bound=plan.upper_bound,
        counts=tuple(counts),
        plan=plan,
    )


def compute_prime_shift_profile(
    request: PrimeShiftRequest,
) -> PrimeShiftProfileResult:
    """Compute the translated-prime representation count for every n in [L, U].

    For each n, count representations n = p + 2^k where p is prime and k >= 0.
    This is the de Polignac-style count: for each prime p and each power of 2
    that satisfies 2^k <= n - p, we count one representation.
    """
    try:
        plan = require_prime_shift_profile_admission(
            request.lower_bound, request.upper_bound
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.translated_prime.admission",
            message=str(exc),
        ) from exc

    return _compute_prime_shift_profile(plan)


def _as_python_integer(value: SupportsIndex | IntegerValue) -> int:
    """Return one direct native integer input as a Python integer."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return operator.index(value)


def prime_shift_profile(
    lower_bound: SupportsIndex | IntegerValue,
    upper_bound: SupportsIndex | IntegerValue,
) -> PrimeShiftProfileResult:
    """Return the exact translated-prime profile on the closed interval [L, U].

    Native callers provide the interval bounds directly.  The operation uses
    the same owner admission plan, segmented-sieve kernel, and typed result
    construction as the MCP adapter.
    """

    lower = _as_python_integer(lower_bound)
    upper = _as_python_integer(upper_bound)
    plan = require_prime_shift_profile_admission(lower, upper)
    return _compute_prime_shift_profile(plan)


__all__ = ["compute_prime_shift_profile", "prime_shift_profile"]
