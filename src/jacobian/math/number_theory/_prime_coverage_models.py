"""Typed contracts for prime-coverage profiles."""

from __future__ import annotations

import math

from pydantic import Field

from jacobian._models import StrictModel

# Coverage rows are materialized as parallel residual/count arrays and typed
# result rows. Bound that allocation-producing cardinality independently of
# any JSON delivery mechanism.
MAX_COVERAGE_UPPER: int = (1 << 53) - 1
MAX_COVERAGE_ROWS: int = 250_000
MAX_COVERAGE_WORK: int = 50_000_000
_MAX_DISTINCT_PRIME_COUNT = 14


def _coverage_work_upper_bound(lower_bound: int, upper_bound: int) -> int:
    """Bound segmented-sieve work without materializing a prefix through U.

    The base sieve costs at most ``sqrt(U) * (log2(sqrt(U)) + 1)`` simple
    steps. For each base prime, the segment contains at most one first-hit
    step plus ``width / p`` hits; summing over all integers gives the
    conservative harmonic bound below.
    """

    width = upper_bound - lower_bound + 1
    root = math.isqrt(upper_bound)
    if root < 2:
        return width
    digit_bound = root.bit_length()
    return root * (digit_bound + 1) + root + width * digit_bound


class PrimeCoverageProfileRequest(StrictModel):
    """A bounded closed interval [L, U] for prime-coverage profiling."""

    lower_bound: int = Field(ge=1, le=MAX_COVERAGE_UPPER)
    upper_bound: int = Field(ge=1, le=MAX_COVERAGE_UPPER)


class PrimeCoverageProfileRow(StrictModel):
    """One (n, omega(n)) pair where omega(n) is the number of distinct prime factors."""

    n: int = Field(ge=1, le=MAX_COVERAGE_UPPER)
    distinct_prime_count: int = Field(ge=0, le=_MAX_DISTINCT_PRIME_COUNT)


class PrimeCoverageProfileResult(StrictModel):
    """Complete ordered prime-coverage table over [L, U]."""

    lower_bound: int
    upper_bound: int
    rows: list[PrimeCoverageProfileRow]


__all__ = [
    "MAX_COVERAGE_ROWS",
    "MAX_COVERAGE_UPPER",
    "MAX_COVERAGE_WORK",
    "PrimeCoverageProfileRequest",
    "PrimeCoverageProfileResult",
    "PrimeCoverageProfileRow",
]
