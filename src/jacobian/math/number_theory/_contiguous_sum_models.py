"""Typed contracts for contiguous-sum representation profiles."""

from __future__ import annotations

from math import isqrt
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

# The segmented regime stores one residual and one divisor count per requested
# integer. The high-magnitude regime factors each requested integer directly,
# so its width is deliberately narrower than the dense interval regime.
MAX_INTERVAL_WIDTH: int = 100_000
MAX_PROFILE_INTEGER_DIGITS: int = 20
MAX_FACTORING_INTERVAL_WIDTH: int = 128
MAX_INTERVAL_WORK: int = 6_000_000
MAX_INTERVAL_RESULT_BYTES: int = 8_000_000
MAX_SEGMENTED_SIEVE_UPPER: int = 10**12
MAX_FACTORING_WORK_SECONDS: int = 60

ContiguousSumInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_PROFILE_INTEGER_DIGITS, strict=True),
]


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
                "max_factoring_work_seconds": MAX_FACTORING_WORK_SECONDS,
            },
        }
    )

    lower_bound: ContiguousSumInteger = Field(
        description=(
            "Inclusive lower endpoint as a canonical positive decimal integer "
            f"with at most {MAX_PROFILE_INTEGER_DIGITS} digits."
        ),
    )
    upper_bound: ContiguousSumInteger = Field(
        description=(
            "Inclusive upper endpoint as a canonical positive decimal integer "
            f"with at most {MAX_PROFILE_INTEGER_DIGITS} digits."
        ),
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        lower = parse_canonical_integer(self.lower_bound)
        upper = parse_canonical_integer(self.upper_bound)
        if lower < 1 or upper < 1:
            raise ValueError("interval endpoints must be positive")
        if upper < lower:
            raise ValueError("upper_bound must be >= lower_bound")
        width = upper - lower + 1
        if width > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        upper_digits = len(self.upper_bound)
        if upper > MAX_SEGMENTED_SIEVE_UPPER:
            if width > MAX_FACTORING_INTERVAL_WIDTH:
                raise ValueError(
                    "high-magnitude intervals exceed the direct-factorization width bound"
                )
            estimated_work = width * upper_digits * 1_000
        else:
            estimated_work = 3 * isqrt(upper) + width * (upper_digits + 1)
        if estimated_work > MAX_INTERVAL_WORK:
            raise ValueError("interval work exceeds the maximum supported budget")
        estimated_result_bytes = width * (upper_digits + 32)
        if estimated_result_bytes > MAX_INTERVAL_RESULT_BYTES:
            raise ValueError("interval result exceeds the maximum supported size")
        return self


class ContiguousSumProfileRow(StrictModel):
    """One (n, count) pair where count is the number of contiguous-sum representations."""

    n: ContiguousSumInteger
    representation_count: StrictInt = Field(ge=1)


class ContiguousSumProfileResult(StrictModel):
    """Complete or operationally incomplete profile over a closed interval."""

    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    lower_bound: ContiguousSumInteger
    upper_bound: ContiguousSumInteger
    rows: tuple[ContiguousSumProfileRow, ...] = Field(
        min_length=0, max_length=MAX_INTERVAL_WIDTH
    )
    detail: str | None = None

    @classmethod
    def _unknown(
        cls,
        *,
        lower_bound: ContiguousSumInteger,
        upper_bound: ContiguousSumInteger,
        detail: str,
    ) -> Self:
        return cls(
            status="UNKNOWN",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rows=(),
            detail=detail,
        )

    @model_validator(mode="after")
    def require_ordered_interval_rows(self) -> Self:
        lower = parse_canonical_integer(self.lower_bound)
        upper = parse_canonical_integer(self.upper_bound)
        if lower < 1 or upper < lower:
            raise ValueError("result endpoints must form a positive interval")
        if self.status == "UNKNOWN":
            if self.rows or not self.detail:
                raise ValueError("an unknown profile has no rows and includes a detail")
            return self
        if self.detail is not None:
            raise ValueError("a complete profile cannot include a detail")
        expected_width = upper - lower + 1
        if len(self.rows) != expected_width:
            raise ValueError("a complete profile has one row per interval integer")
        for expected, row in zip(range(lower, upper + 1), self.rows, strict=True):
            if parse_canonical_integer(row.n) != expected:
                raise ValueError("profile rows must be ordered over the interval")
        return self


__all__ = [
    "MAX_FACTORING_INTERVAL_WIDTH",
    "MAX_FACTORING_WORK_SECONDS",
    "MAX_INTERVAL_RESULT_BYTES",
    "MAX_INTERVAL_WIDTH",
    "MAX_INTERVAL_WORK",
    "MAX_PROFILE_INTEGER_DIGITS",
    "MAX_SEGMENTED_SIEVE_UPPER",
    "ContiguousSumInteger",
    "ContiguousSumProfileRequest",
    "ContiguousSumProfileResult",
    "ContiguousSumProfileRow",
]
