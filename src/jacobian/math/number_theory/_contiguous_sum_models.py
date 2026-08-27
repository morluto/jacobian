"""Typed contracts for contiguous-sum representation profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_INTERVAL_UPPER: int = 1_000_000
MAX_INTERVAL_WIDTH: int = 100_000


class ContiguousSumProfileRequest(StrictModel):
    """A bounded closed positive interval [L, U] for contiguous-sum profiling."""

    lower_bound: int = Field(ge=1, le=MAX_INTERVAL_UPPER)
    upper_bound: int = Field(ge=1, le=MAX_INTERVAL_UPPER)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.upper_bound - self.lower_bound + 1 > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        return self


class ContiguousSumProfileRow(StrictModel):
    """One (n, count) pair where count is the number of contiguous-sum representations."""

    n: int
    representation_count: int = Field(ge=1)


class ContiguousSumProfileResult(StrictModel):
    """Complete ordered contiguous-sum representation table over [L, U]."""

    lower_bound: int
    upper_bound: int
    rows: list[ContiguousSumProfileRow]


__all__ = [
    "MAX_INTERVAL_UPPER",
    "MAX_INTERVAL_WIDTH",
    "ContiguousSumProfileRequest",
    "ContiguousSumProfileResult",
    "ContiguousSumProfileRow",
]
