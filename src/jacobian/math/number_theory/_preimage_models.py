"""Typed contracts for k*sigma(k) preimage and p-adic interval valuation profiles."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing import Self

from jacobian._models import StrictModel

MAX_PREIMAGE_K: int = 1000
MAX_VALUATION_UPPER: int = 10_000_000
MAX_VALUATION_WIDTH: int = 1_000_000


class KSigmaPreimageRequest(StrictModel):
    """Find all n such that k*sigma(n) = target_value."""

    k: int = Field(ge=1, le=100)
    target_value: int = Field(ge=1, le=10_000_000)


class KSigmaPreimageResult(StrictModel):
    """All preimages n such that k*sigma(n) = target_value."""

    k: int
    target_value: int
    preimages: list[int]


class IntervalValuationProfileRequest(StrictModel):
    """A bounded closed interval [L, U] and a prime p for p-adic valuation profiling."""

    lower_bound: int = Field(ge=1, le=MAX_VALUATION_UPPER)
    upper_bound: int = Field(ge=1, le=MAX_VALUATION_UPPER)
    prime: int = Field(ge=2, le=10000)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.upper_bound - self.lower_bound + 1 > MAX_VALUATION_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        from sympy import isprime
        if not isprime(self.prime):
            raise ValueError("prime must be a prime number")
        return self


class IntervalValuationProfileRow(StrictModel):
    """One (n, v_p(n)) pair in a p-adic valuation profile."""

    n: int
    valuation: int = Field(ge=0)


class IntervalValuationProfileResult(StrictModel):
    """Complete p-adic valuation profile over [L, U]."""

    lower_bound: int
    upper_bound: int
    prime: int
    rows: list[IntervalValuationProfileRow]


__all__ = [
    "KSigmaPreimageRequest",
    "KSigmaPreimageResult",
    "IntervalValuationProfileRequest",
    "IntervalValuationProfileResult",
    "IntervalValuationProfileRow",
    "MAX_VALUATION_UPPER",
    "MAX_VALUATION_WIDTH",
]
