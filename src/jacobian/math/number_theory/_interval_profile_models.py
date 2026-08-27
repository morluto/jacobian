"""Typed contracts for bounded integer-interval arithmetic-function profiles."""

from __future__ import annotations

import math
from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)

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


def _json_array_size(item_size: int, count: int) -> int:
    return 2 + max(count - 1, 0) + count * item_size


def _interval_work_upper_bound(lower_bound: int, upper_bound: int) -> int:
    """Bound base-sieve and interval-hit work without a prefix through U."""

    width = upper_bound - lower_bound + 1
    root = math.isqrt(upper_bound)
    if root < 2:
        return width
    digit_bound = root.bit_length()
    return root * (digit_bound + 1) + root + width * digit_bound


def _squarefree_result_upper_bound_bytes(lower_bound: int, upper_bound: int) -> int:
    """Bound the compact partition result using its two integer arrays."""

    width = upper_bound - lower_bound + 1
    integer_size = len(encode_strict_json(upper_bound))
    # The two arrays partition the interval, so their combined value and
    # separator cost is bounded by one array's values plus two brackets and
    # one separator per possible entry.
    partition_arrays_size = 4 + width + 1 + width * integer_size
    count_size = len(encode_strict_json(width))
    return strict_json_object_size(
        (
            ("lower_bound", len(encode_strict_json(lower_bound))),
            ("nonsquarefree_values", partition_arrays_size),
            ("nonsquarefree_count", count_size),
            ("squarefree_values", 0),
            ("squarefree_count", count_size),
            ("upper_bound", len(encode_strict_json(upper_bound))),
        )
    )


def _row_profile_result_upper_bound_bytes(lower_bound: int, upper_bound: int) -> int:
    """Bound a row profile using its widest field and safe value envelope."""

    width = upper_bound - lower_bound + 1
    row_size = strict_json_object_size(
        (
            ("greatest_prime_factor", len(encode_strict_json(upper_bound**2))),
            ("n", len(encode_strict_json(upper_bound))),
        )
    )
    return strict_json_object_size(
        (
            ("lower_bound", len(encode_strict_json(lower_bound))),
            ("rows", _json_array_size(row_size, width)),
            ("upper_bound", len(encode_strict_json(upper_bound))),
        )
    )


def _prime_gap_result_upper_bound_bytes(lower_bound: int, upper_bound: int) -> int:
    """Bound prime-gap rows using a global prime-count envelope."""

    width = upper_bound - lower_bound + 1
    if upper_bound < 2:
        row_count = 0
    elif upper_bound < 17:
        row_count = width
    else:
        # pi(x) < 2x/log(x) for x >= 2; using pi(U) bounds every prime in
        # [L,U]. The factor of two also bounds the successor and its gap.
        row_count = min(width, math.ceil(2 * upper_bound / math.log(upper_bound)))
    row_size = strict_json_object_size(
        (
            ("gap", len(encode_strict_json(2 * upper_bound))),
            ("lower_prime", len(encode_strict_json(upper_bound))),
            ("upper_prime", len(encode_strict_json(2 * upper_bound))),
        )
    )
    return strict_json_object_size(
        (
            ("lower_bound", len(encode_strict_json(lower_bound))),
            ("rows", _json_array_size(row_size, row_count)),
            ("upper_bound", len(encode_strict_json(upper_bound))),
        )
    )


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
            "and the bounded profile work envelope."
        ),
    )

    @model_validator(mode="after")
    def require_admitted_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.width() > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        if (
            _interval_work_upper_bound(self.lower_bound, self.upper_bound)
            > MAX_SIEVE_WORK
        ):
            raise ValueError(
                "interval exceeds the segmented profile work budget of "
                f"{MAX_SIEVE_WORK} steps"
            )
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
            _interval_work_upper_bound(self.lower_bound, self.upper_bound)
            <= MAX_SIEVE_WORK
        )


class SquarefreeProfileRequest(IntervalProfileRequest):
    """Interval request admitted against the compact squarefree result."""

    @model_validator(mode="after")
    def require_squarefree_output_admission(self) -> Self:
        if (
            _squarefree_result_upper_bound_bytes(self.lower_bound, self.upper_bound)
            > MAX_PROFILE_RESULT_BYTES
        ):
            raise ValueError(
                "squarefree interval result exceeds the canonical output budget"
            )
        return self


class IntervalProfileRowsRequest(IntervalProfileRequest):
    """Interval request admitted against a complete row-profile result."""

    @model_validator(mode="after")
    def require_row_output_admission(self) -> Self:
        if (
            _row_profile_result_upper_bound_bytes(self.lower_bound, self.upper_bound)
            > MAX_PROFILE_RESULT_BYTES
        ):
            raise ValueError("interval row profile exceeds the canonical output budget")
        return self


class PrimeGapProfileRequest(IntervalProfileRequest):
    """Interval request admitted against the sparse prime-gap result."""

    @model_validator(mode="after")
    def require_prime_gap_output_admission(self) -> Self:
        if (
            _prime_gap_result_upper_bound_bytes(self.lower_bound, self.upper_bound)
            > MAX_PROFILE_RESULT_BYTES
        ):
            raise ValueError(
                "prime-gap interval result exceeds the canonical output budget"
            )
        return self


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
    "IntervalProfileRowsRequest",
    "LeastPrimeFactorProfileResult",
    "LeastPrimeFactorProfileRow",
    "PrimeGapProfileRequest",
    "PrimeGapProfileResult",
    "PrimeGapProfileRow",
    "SquarefreeProfileRequest",
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
