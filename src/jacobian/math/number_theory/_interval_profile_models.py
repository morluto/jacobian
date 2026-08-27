"""Typed contracts for bounded integer-interval arithmetic-function profiles."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

# ---------------------------------------------------------------------------
# Admission envelope
# ---------------------------------------------------------------------------
#
# The profiles share one admission shape: a bounded closed interval [L, U]
# with L >= 1 and U >= L.  The key quantities controlling work and output are
# the interval width W = U - L + 1 and the upper bound U (for the base sieve
# through sqrt(U)).  We cap W at a value that keeps the segment and the
# serialized result within the canonical transport limit.
#
# For squarefree and prime-factor profiles the kernels use a segment over
# [L, U] and a base sieve needing primes through floor(sqrt(U)); they never
# materialize a prefix sieve through U.  The work is proportional to the
# segment width and the base sieve plus the interval's prime-factor hits.
#
# For prime-gap profiles the kernel marks primes in [L, U] as a segment and
# queries the successor beyond U separately, needing primes through
# floor(sqrt(U)).

MAX_INTERVAL_UPPER_BOUND: int = 10_000_000
MAX_INTERVAL_WIDTH: int = 1_000_000
MAX_SIEVE_WORK: int = 20_000_000
MAX_PROFILE_RESULT_BYTES: int = CanonicalLimits().max_output_bytes
# With U <= 10 million, every row/list entry is below 64 canonical bytes;
# reserve fixed envelope space for the result object and its interval fields.
_PROFILE_ROW_BYTES: int = 64
_PROFILE_RESULT_ENVELOPE_BYTES: int = 256


class IntervalProfileRequest(StrictModel):
    """A bounded closed interval [L, U] with 1 <= L <= U."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A closed interval with 1 <= lower_bound <= upper_bound. "
                "The interval width and the complete profile result must fit "
                "the bounded execution and canonical-output envelope."
            )
        }
    )

    lower_bound: StrictInt = Field(
        ge=1,
        le=MAX_INTERVAL_UPPER_BOUND,
        description="Inclusive lower endpoint of the interval.",
    )
    upper_bound: StrictInt = Field(
        ge=1,
        le=MAX_INTERVAL_UPPER_BOUND,
        description=(
            "Inclusive upper endpoint; it must be at least lower_bound, "
            "and the complete result must fit the output budget."
        ),
    )

    @model_validator(mode="after")
    def require_admitted_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.width() > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        if (
            _PROFILE_RESULT_ENVELOPE_BYTES + self.width() * _PROFILE_ROW_BYTES
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
        if self.upper_bound > MAX_INTERVAL_UPPER_BOUND:
            return False
        return (
            _PROFILE_RESULT_ENVELOPE_BYTES + width * _PROFILE_ROW_BYTES
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
    "MAX_PROFILE_RESULT_BYTES",
    "DivisorCountProfileResult",
    "DivisorCountProfileRow",
    "DivisorSumProfileResult",
    "DivisorSumProfileRow",
    "EulerTotientProfileResult",
    "EulerTotientProfileRow",
    "GreatestPrimeFactorProfileResult",
    "GreatestPrimeFactorProfileRow",
    "IntervalProfileRequest",
    "LeastPrimeFactorProfileResult",
    "LeastPrimeFactorProfileRow",
    "PrimeGapProfileResult",
    "PrimeGapProfileRow",
    "SquarefreeProfileResult",
]


class LeastPrimeFactorProfileRow(StrictModel):
    n: int
    least_prime_factor: int = Field(ge=1)


class LeastPrimeFactorProfileResult(StrictModel):
    lower_bound: int
    upper_bound: int
    rows: tuple[LeastPrimeFactorProfileRow, ...]


class EulerTotientProfileRow(StrictModel):
    n: int
    euler_totient: int = Field(ge=1)


class EulerTotientProfileResult(StrictModel):
    lower_bound: int
    upper_bound: int
    rows: tuple[EulerTotientProfileRow, ...]


class DivisorSumProfileRow(StrictModel):
    n: int
    divisor_sum: int = Field(ge=1)


class DivisorSumProfileResult(StrictModel):
    lower_bound: int
    upper_bound: int
    rows: tuple[DivisorSumProfileRow, ...]
