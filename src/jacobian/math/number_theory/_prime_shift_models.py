"""Typed contracts for translated-prime representation profiles."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing import Self

from jacobian._models import StrictModel

MAX_SHIFT_INTERVAL_UPPER: int = 10_000_000
MAX_SHIFT_INTERVAL_WIDTH: int = 1_000_000


class PrimeShiftProfileRequest(StrictModel):
    """A bounded closed interval [L, U] for translated-prime representation counting."""

    lower_bound: int = Field(ge=1, le=MAX_SHIFT_INTERVAL_UPPER)
    upper_bound: int = Field(ge=1, le=MAX_SHIFT_INTERVAL_UPPER)

    @model_validator(mode="after")
    def require_valid_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.upper_bound - self.lower_bound + 1 > MAX_SHIFT_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        return self


class PrimeShiftProfileRow(StrictModel):
    """One (n, count) pair where count is the number of representations n = p + 2^k."""

    n: int
    representation_count: int = Field(ge=0)


class PrimeShiftProfileResult(StrictModel):
    """Complete ordered translated-prime representation table over [L, U]."""

    lower_bound: int
    upper_bound: int
    rows: list[PrimeShiftProfileRow]


__all__ = [
    "PrimeShiftProfileRequest",
    "PrimeShiftProfileResult",
    "PrimeShiftProfileRow",
    "MAX_SHIFT_INTERVAL_UPPER",
    "MAX_SHIFT_INTERVAL_WIDTH",
]
