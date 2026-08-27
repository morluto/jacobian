"""Typed contracts for contiguous-sum representation profiles."""

from __future__ import annotations

from math import isqrt
from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._models import StrictModel

# The segmented regime stores one residual and one divisor count per requested
# integer. The high-magnitude regime factors each requested integer directly,
# so its width is deliberately narrower than the dense interval regime.
MAX_INTERVAL_WIDTH: int = 100_000
MAX_PROFILE_INTEGER_DIGITS: int = 20
MAX_FACTORING_INTERVAL_WIDTH: int = 128
MAX_INTERVAL_WORK: int = 6_000_000
MAX_INTERVAL_RESULT_BYTES: int = 8_000_000
MAX_SEGMENTED_SIEVE_UPPER: int = 10**12


class ContiguousSumProfileRequest(StrictModel):
    """A bounded closed positive interval [L, U] for contiguous-sum profiling.

    Endpoints are strict positive integers of at most 20 decimal digits. The
    interval contains at most 100,000 integers; intervals above 10**12 use
    direct factorization and therefore contain at most 128 integers.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A closed positive integer interval [lower_bound, upper_bound]. "
                f"Endpoints must be strict integers with at most "
                f"{MAX_PROFILE_INTEGER_DIGITS} decimal digits. The interval "
                f"contains at most {MAX_INTERVAL_WIDTH:,} integers; intervals "
                f"above {MAX_SEGMENTED_SIEVE_UPPER:,} use direct factorization "
                f"and contain at most {MAX_FACTORING_INTERVAL_WIDTH} integers."
            ),
            "x-jacobian-bounds": {
                "max_interval_width": MAX_INTERVAL_WIDTH,
                "max_profile_integer_digits": MAX_PROFILE_INTEGER_DIGITS,
                "max_factoring_interval_width": MAX_FACTORING_INTERVAL_WIDTH,
                "max_interval_work": MAX_INTERVAL_WORK,
                "max_interval_result_bytes": MAX_INTERVAL_RESULT_BYTES,
                "segmented_sieve_upper": MAX_SEGMENTED_SIEVE_UPPER,
            },
        }
    )

    lower_bound: StrictInt = Field(
        ge=1,
        description="Inclusive lower endpoint; a strict positive integer.",
    )
    upper_bound: StrictInt = Field(
        ge=1,
        description="Inclusive upper endpoint; a strict positive integer.",
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        width = self.upper_bound - self.lower_bound + 1
        if width > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        upper_digits = len(str(self.upper_bound))
        if upper_digits > MAX_PROFILE_INTEGER_DIGITS:
            raise ValueError(
                "interval endpoints exceed the maximum supported decimal digit length"
            )
        if self.upper_bound > MAX_SEGMENTED_SIEVE_UPPER:
            if width > MAX_FACTORING_INTERVAL_WIDTH:
                raise ValueError(
                    "high-magnitude intervals exceed the direct-factorization width bound"
                )
            estimated_work = width * upper_digits * 1_000
        else:
            estimated_work = 3 * isqrt(self.upper_bound) + width * (upper_digits + 1)
        if estimated_work > MAX_INTERVAL_WORK:
            raise ValueError("interval work exceeds the maximum supported budget")
        estimated_result_bytes = width * (upper_digits + 32)
        if estimated_result_bytes > MAX_INTERVAL_RESULT_BYTES:
            raise ValueError("interval result exceeds the maximum supported size")
        return self


class ContiguousSumProfileRow(StrictModel):
    """One (n, count) pair where count is the number of contiguous-sum representations."""

    n: StrictInt
    representation_count: StrictInt = Field(ge=1)


class ContiguousSumProfileResult(StrictModel):
    """Complete ordered contiguous-sum representation table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: list[ContiguousSumProfileRow]


__all__ = [
    "MAX_FACTORING_INTERVAL_WIDTH",
    "MAX_INTERVAL_RESULT_BYTES",
    "MAX_INTERVAL_WIDTH",
    "MAX_INTERVAL_WORK",
    "MAX_PROFILE_INTEGER_DIGITS",
    "MAX_SEGMENTED_SIEVE_UPPER",
    "ContiguousSumProfileRequest",
    "ContiguousSumProfileResult",
    "ContiguousSumProfileRow",
]
