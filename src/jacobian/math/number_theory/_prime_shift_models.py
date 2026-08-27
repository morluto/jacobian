"""Typed contracts for translated-prime representation profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)

MAX_SHIFT_INTERVAL_UPPER: int = 10_000_000
MAX_SHIFT_INTERVAL_WIDTH: int = 1_000_000
MAX_SHIFT_RESULT_BYTES: int = CanonicalLimits().max_output_bytes


def _profile_result_byte_bound(lower_bound: int, upper_bound: int) -> int:
    """Bound the complete canonical JSON result before materializing rows.

    Each row contains one interval value and one count. A count cannot exceed
    the number of powers of two no larger than ``upper_bound - 2`` because a
    prime summand is at least two. Charging every row at the widest possible
    value and count is conservative while preserving the exact result shape.
    """
    width = upper_bound - lower_bound + 1
    maximum_count = (upper_bound - 2).bit_length() if upper_bound >= 3 else 0
    row_bytes = strict_json_object_size(
        (
            ("n", len(encode_strict_json(upper_bound))),
            ("representation_count", len(encode_strict_json(maximum_count))),
        )
    )
    rows_bytes = 2 + max(width - 1, 0) + width * row_bytes
    return strict_json_object_size(
        (
            ("lower_bound", len(encode_strict_json(lower_bound))),
            ("upper_bound", len(encode_strict_json(upper_bound))),
            ("rows", rows_bytes),
        )
    )


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
        if (
            _profile_result_byte_bound(self.lower_bound, self.upper_bound)
            > MAX_SHIFT_RESULT_BYTES
        ):
            raise ValueError("interval result exceeds the canonical output budget")
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
    "MAX_SHIFT_INTERVAL_UPPER",
    "MAX_SHIFT_INTERVAL_WIDTH",
    "MAX_SHIFT_RESULT_BYTES",
    "PrimeShiftProfileRequest",
    "PrimeShiftProfileResult",
    "PrimeShiftProfileRow",
]
