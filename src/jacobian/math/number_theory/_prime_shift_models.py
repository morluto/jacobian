"""Typed contracts for translated-prime representation profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_SHIFT_INTERVAL_WIDTH: int = 1_000_000
MAX_SHIFT_WORK: int = 100_000_000
_MAX_SHIFT_MARK_WORK_MULTIPLIER: int = 16


@dataclass(frozen=True, slots=True)
class _PrimeShiftProfileExecutionPlan:
    """One admitted profile's complete, request-scoped execution envelope."""

    lower_bound: int
    upper_bound: int
    base_limit: int
    candidate_intervals: tuple[tuple[int, int, int], ...]
    candidate_work: int


def _validate_prime_shift_interval(lower_bound: int, upper_bound: int) -> None:
    """Validate the structural interval shared by request and result values."""

    if lower_bound < 1:
        raise ValueError("lower_bound must be >= 1")
    if upper_bound < 1:
        raise ValueError("upper_bound must be >= 1")
    if upper_bound < lower_bound:
        raise ValueError("upper_bound must be >= lower_bound")


class PrimeShiftProfileRequest(StrictModel):
    """A bounded closed interval [L, U] for translated-prime representation counting."""

    lower_bound: int = Field(ge=1)
    upper_bound: int = Field(ge=1)


def require_prime_shift_profile_admission(
    lower_bound: int,
    upper_bound: int,
) -> _PrimeShiftProfileExecutionPlan:
    """Build one exact execution plan after structural request validation.

    The base-prime sieve and every power-specific candidate interval are
    charged before execution.  The derived base-sieve charge also bounds the
    endpoint size, so a narrow interval can use larger integers when its
    actual segmented-sieve work and complete result still fit.
    """
    _validate_prime_shift_interval(lower_bound, upper_bound)
    if upper_bound - lower_bound + 1 > MAX_SHIFT_INTERVAL_WIDTH:
        raise ValueError("interval width exceeds maximum supported width")
    maximum_power = upper_bound - 2
    power_count = maximum_power.bit_length() if maximum_power >= 1 else 0
    base_work_units = power_count + 1
    max_base_limit_plus_one = MAX_SHIFT_WORK // (
        _MAX_SHIFT_MARK_WORK_MULTIPLIER * base_work_units
    )
    if max_base_limit_plus_one == 0 or upper_bound >= max_base_limit_plus_one**2:
        raise ValueError("interval exceeds the translated-prime work budget")

    base_limit = math.isqrt(upper_bound)
    candidate_intervals: list[tuple[int, int, int]] = []
    candidate_count = 0
    power = 1
    while power <= maximum_power:
        candidate_lower = max(2, lower_bound - power)
        candidate_upper = upper_bound - power
        if candidate_lower <= candidate_upper:
            candidate_intervals.append((power, candidate_lower, candidate_upper))
            candidate_count += candidate_upper - candidate_lower + 1
        power <<= 1

    candidate_work = _MAX_SHIFT_MARK_WORK_MULTIPLIER * (
        candidate_count + (len(candidate_intervals) + 1) * (base_limit + 1)
    )
    if candidate_work > MAX_SHIFT_WORK:
        raise ValueError("interval exceeds the translated-prime work budget")
    return _PrimeShiftProfileExecutionPlan(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        base_limit=base_limit,
        candidate_intervals=tuple(candidate_intervals),
        candidate_work=candidate_work,
    )


class PrimeShiftProfileRow(StrictModel):
    """One (n, count) pair where count is the number of representations n = p + 2^k."""

    n: int
    representation_count: int = Field(ge=0)


class PrimeShiftProfileResult(StrictModel):
    """Complete ordered translated-prime representation table over [L, U]."""

    lower_bound: int
    upper_bound: int
    rows: tuple[PrimeShiftProfileRow, ...] = Field(max_length=MAX_SHIFT_INTERVAL_WIDTH)

    @model_validator(mode="after")
    def bind_row_axis(self) -> Self:
        _validate_prime_shift_interval(self.lower_bound, self.upper_bound)
        width = self.upper_bound - self.lower_bound + 1
        if len(self.rows) != width or any(
            row.n != self.lower_bound + offset for offset, row in enumerate(self.rows)
        ):
            raise ValueError(
                "rows must contain exactly one consecutive n for every value "
                "in the declared interval"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        lower_bound: int,
        upper_bound: int,
        counts: tuple[int, ...],
        plan: _PrimeShiftProfileExecutionPlan,
    ) -> Self:
        """Construct the trusted result from the plan's admitted row axis."""
        if (
            lower_bound != plan.lower_bound
            or upper_bound != plan.upper_bound
            or len(counts) != plan.upper_bound - plan.lower_bound + 1
        ):
            raise RuntimeError("prime-shift kernel result does not match its plan")
        return cls.model_construct(
            lower_bound=plan.lower_bound,
            upper_bound=plan.upper_bound,
            rows=tuple(
                PrimeShiftProfileRow(
                    n=plan.lower_bound + index,
                    representation_count=count,
                )
                for index, count in enumerate(counts)
            ),
        )


__all__ = [
    "MAX_SHIFT_INTERVAL_WIDTH",
    "MAX_SHIFT_WORK",
    "PrimeShiftProfileRequest",
    "PrimeShiftProfileResult",
    "PrimeShiftProfileRow",
]
