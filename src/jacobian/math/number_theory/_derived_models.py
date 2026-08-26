"""Contracts owned by the derived integer kernels."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import _validation_error


class FloorSquareRootRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=1_000_000_000_000)


class FloorSquareRootResult(StrictModel):
    """The exact floor of the nonnegative integer square root."""

    root: StrictInt = Field(ge=0, le=1_000_000)


def _is_bounded_prime(value: int) -> bool:
    """Decide primality within the Legendre-denominator admission envelope."""

    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    candidate = 5
    while candidate * candidate <= value:
        if value % candidate == 0 or value % (candidate + 2) == 0:
            return False
        candidate += 6
    return True


class LegendreSymbolRequest(StrictModel):
    """Arguments for the Legendre symbol with a bounded odd prime denominator."""

    a: StrictInt = Field(ge=-(2**53 - 1), le=2**53 - 1)
    prime: StrictInt = Field(ge=3, le=10_000_000)

    @model_validator(mode="after")
    def require_prime_denominator(self) -> Self:
        if not _is_bounded_prime(self.prime):
            raise _validation_error(
                "legendre_denominator_must_be_prime",
                "Legendre denominator must be prime",
            )
        return self


class LegendreSymbolResult(StrictModel):
    a: StrictInt
    prime: StrictInt = Field(ge=3, le=10_000_000)
    symbol: Literal[-1, 0, 1]


class FactorialValuationRequest(StrictModel):
    """Arguments for the largest exponent ``e`` such that ``base**e`` divides ``n!``."""

    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)


class FactorialValuationResult(StrictModel):
    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)
    valuation: StrictInt = Field(ge=0)


__all__ = [
    "FactorialValuationRequest",
    "FactorialValuationResult",
    "FloorSquareRootRequest",
    "FloorSquareRootResult",
    "LegendreSymbolRequest",
    "LegendreSymbolResult",
]
