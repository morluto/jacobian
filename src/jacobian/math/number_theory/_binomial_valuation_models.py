"""Typed contracts for p-adic valuation profiles of binomial coefficients."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel

MAX_BINOMIAL_PROFILE_ROWS = 500_000
MAX_BINOMIAL_DIGIT_WORK = 2_000_000
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def _base_digit_count(value: int, base: int) -> int:
    """Return the number of base-``base`` digits in a nonnegative value."""

    if value == 0:
        return 0
    digits = 0
    while value:
        value //= base
        digits += 1
    return digits


class BinomialValuationProfileRequest(StrictModel):
    """Parameters for computing v_p(C(n,k)) for all k from 0 to n."""

    n: int = Field(ge=0)
    prime: int = Field(ge=2, le=_MAX_SAFE_JSON_INTEGER)


class BinomialValuationProfileRow(StrictModel):
    """One (k, v_p(C(n,k))) pair."""

    k: int = Field(ge=0)
    valuation: int = Field(ge=0)


class BinomialValuationProfileResult(StrictModel):
    """Complete v_p(C(n,k)) profile for k=0..n."""

    n: int
    prime: int
    rows: list[BinomialValuationProfileRow]


__all__ = [
    "MAX_BINOMIAL_DIGIT_WORK",
    "MAX_BINOMIAL_PROFILE_ROWS",
    "BinomialValuationProfileRequest",
    "BinomialValuationProfileResult",
    "BinomialValuationProfileRow",
]
