"""Typed contracts for bounded integer-interval arithmetic-function profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

# ---------------------------------------------------------------------------
# Admission envelope
# ---------------------------------------------------------------------------
#
# The profiles share one admission shape: a bounded closed interval [L, U]
# with L >= 1 and U >= L.  The key quantities controlling work and output are
# the interval width W = U - L + 1 and the upper bound U (for the base sieve
# through sqrt(U)).  The result estimate is deliberately conservative: every
# interval value is charged 64 bytes plus fixed envelope overhead.
#
# For squarefree/divisor-count/greatest-prime-factor profiles the kernel is a
# segmented sieve over [L, U] needing primes through floor(sqrt(U)).
#
# For prime-gap profiles the kernel is a segmented prime sieve over [L, U],
# followed by one successor-prime query when the interval contains a prime.

MAX_INTERVAL_UPPER_BOUND: int = 10_000_000
MAX_INTERVAL_WIDTH: int = 1_000_000
MAX_SIEVE_WORK: int = 20_000_000
MAX_PROFILE_RESULT_BYTES: int = CanonicalLimits().max_output_bytes
_PROFILE_ROW_BYTES: int = 64
_PROFILE_RESULT_OVERHEAD_BYTES: int = 1_024


class IntervalProfileRequest(StrictModel):
    """A bounded closed interval [L, U] with 1 <= L <= U."""

    lower_bound: StrictInt = Field(ge=1, le=MAX_INTERVAL_UPPER_BOUND)
    upper_bound: StrictInt = Field(ge=1, le=MAX_INTERVAL_UPPER_BOUND)

    @model_validator(mode="after")
    def require_admitted_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.width() > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        if (
            self.width() * _PROFILE_ROW_BYTES + _PROFILE_RESULT_OVERHEAD_BYTES
            > MAX_PROFILE_RESULT_BYTES
        ):
            raise ValueError("interval result exceeds the canonical output budget")
        return self

    def width(self) -> int:
        return self.upper_bound - self.lower_bound + 1

    def is_admitted(self) -> bool:
        width = self.width()
        if width < 1:
            return False
        if width > MAX_INTERVAL_WIDTH:
            return False
        return (
            self.upper_bound <= MAX_INTERVAL_UPPER_BOUND
            and width * _PROFILE_ROW_BYTES + _PROFILE_RESULT_OVERHEAD_BYTES
            <= MAX_PROFILE_RESULT_BYTES
        )


# ---------------------------------------------------------------------------
# Squarefree profile result
# ---------------------------------------------------------------------------


class SquarefreeProfileResult(StrictModel):
    """Complete exact squarefree/non-squarefree partition of [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    squarefree_values: tuple[StrictInt, ...]
    nonsquarefree_values: tuple[StrictInt, ...]
    squarefree_count: StrictInt
    nonsquarefree_count: StrictInt


# ---------------------------------------------------------------------------
# Divisor-count profile result
# ---------------------------------------------------------------------------


class DivisorCountProfileRow(StrictModel):
    """One (n, tau(n)) pair in a divisor-count profile."""

    n: StrictInt
    divisor_count: StrictInt = Field(ge=1)


class DivisorCountProfileResult(StrictModel):
    """Complete ordered divisor-count table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[DivisorCountProfileRow, ...]


# ---------------------------------------------------------------------------
# Greatest-prime-factor profile result
# ---------------------------------------------------------------------------


class GreatestPrimeFactorProfileRow(StrictModel):
    """One (n, P+(n)) pair in a greatest-prime-factor profile."""

    n: StrictInt
    greatest_prime_factor: StrictInt = Field(ge=1)


class GreatestPrimeFactorProfileResult(StrictModel):
    """Complete ordered greatest-prime-factor table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[GreatestPrimeFactorProfileRow, ...]


# ---------------------------------------------------------------------------
# Prime-gap profile result
# ---------------------------------------------------------------------------


class PrimeGapProfileRow(StrictModel):
    """One consecutive-prime pair (p, q, q - p) in a prime-gap profile."""

    lower_prime: StrictInt = Field(ge=2)
    upper_prime: StrictInt = Field(ge=2)
    gap: StrictInt = Field(ge=1)


class PrimeGapProfileResult(StrictModel):
    """Complete ordered consecutive-prime gap table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[PrimeGapProfileRow, ...]


__all__ = [
    "MAX_INTERVAL_UPPER_BOUND",
    "MAX_INTERVAL_WIDTH",
    "DivisorCountProfileResult",
    "DivisorCountProfileRow",
    "GreatestPrimeFactorProfileResult",
    "GreatestPrimeFactorProfileRow",
    "IntervalProfileRequest",
    "PrimeGapProfileResult",
    "PrimeGapProfileRow",
    "SquarefreeProfileResult",
]
