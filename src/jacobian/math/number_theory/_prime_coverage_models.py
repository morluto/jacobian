"""Typed contracts for prime-coverage profiles."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)

# Coverage rows use JSON integers, so this is a transport-derived source
# boundary rather than a sieve boundary. Actual execution is admitted by the
# square-root work estimate below.
MAX_COVERAGE_UPPER: int = (1 << 53) - 1
MAX_COVERAGE_RESULT_BYTES: int = CanonicalLimits().max_output_bytes
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


def _json_array_size(item_size: int, count: int) -> int:
    return 2 + max(count - 1, 0) + count * item_size


def _coverage_result_upper_bound_bytes(lower_bound: int, upper_bound: int) -> int:
    """Bound the exact canonical size of one complete coverage result.

    Every emitted ``n`` is at most ``upper_bound`` and the kernel can produce
    at most fourteen distinct prime factors for values up to ``MAX_COVERAGE_UPPER``.
    The field and array sizes are calculated with the same canonical encoder
    used by the final result boundary, so accepted requests cannot fail only
    during dispatch serialization.
    """

    width = upper_bound - lower_bound + 1
    row_size = strict_json_object_size(
        (
            ("n", len(encode_strict_json(upper_bound))),
            (
                "distinct_prime_count",
                len(encode_strict_json(_MAX_DISTINCT_PRIME_COUNT)),
            ),
        )
    )
    rows_size = _json_array_size(row_size, width)
    return strict_json_object_size(
        (
            ("lower_bound", len(encode_strict_json(lower_bound))),
            ("upper_bound", len(encode_strict_json(upper_bound))),
            ("rows", rows_size),
        )
    )


class PrimeCoverageProfileRequest(StrictModel):
    """A bounded closed interval [L, U] for prime-coverage profiling."""

    lower_bound: int = Field(ge=1, le=MAX_COVERAGE_UPPER)
    upper_bound: int = Field(ge=1, le=MAX_COVERAGE_UPPER)

    @model_validator(mode="after")
    def require_valid_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        predicted = _coverage_result_upper_bound_bytes(
            self.lower_bound, self.upper_bound
        )
        if predicted > MAX_COVERAGE_RESULT_BYTES:
            raise ValueError(
                "interval result exceeds the canonical output budget of "
                f"{MAX_COVERAGE_RESULT_BYTES} bytes"
            )
        work = _coverage_work_upper_bound(self.lower_bound, self.upper_bound)
        if work > MAX_COVERAGE_WORK:
            raise ValueError(
                "interval exceeds the segmented prime-coverage work budget of "
                f"{MAX_COVERAGE_WORK} steps"
            )
        return self


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
    "MAX_COVERAGE_RESULT_BYTES",
    "MAX_COVERAGE_UPPER",
    "MAX_COVERAGE_WORK",
    "PrimeCoverageProfileRequest",
    "PrimeCoverageProfileResult",
    "PrimeCoverageProfileRow",
]
