"""Contracts owned by the derived integer kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._integer_models import MAX_SAFE_INTEGER

MAX_VALUATION_ARGUMENT_DIGITS = 4_096
MAX_FACTORIAL_BASE = 1_000_000
MAX_BINOMIAL_VALUATION_PRIME = 2**64 - 1
MAX_FLOOR_SQUARE_ROOT = isqrt(MAX_SAFE_INTEGER)
MAX_LEGENDRE_PRIME = MAX_SAFE_INTEGER


class FloorSquareRootRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=MAX_SAFE_INTEGER)


class FloorSquareRootResult(StrictModel):
    """The exact floor of the nonnegative integer square root."""

    root: StrictInt = Field(ge=0, le=MAX_FLOOR_SQUARE_ROOT)


class LegendreSymbolRequest(StrictModel):
    """Arguments for the Legendre symbol with a bounded odd prime denominator."""

    a: StrictInt = Field(ge=-MAX_SAFE_INTEGER, le=MAX_SAFE_INTEGER)
    prime: StrictInt = Field(ge=3, le=MAX_LEGENDRE_PRIME)


class LegendreSymbolResult(StrictModel):
    a: StrictInt
    prime: StrictInt = Field(ge=3, le=MAX_LEGENDRE_PRIME)
    symbol: Literal[-1, 0, 1]


class FactorialValuationRequest(StrictModel):
    """Arguments for the largest exponent ``e`` such that ``base**e`` divides ``n!``."""

    n: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    base: CanonicalInteger = Field(max_length=len(str(MAX_FACTORIAL_BASE)))

    @cached_property
    def admitted(self) -> _FactorialValuationInput:
        n = parse_canonical_integer(self.n)
        if n < 0:
            raise PydanticCustomError(
                "number_theory.factorial_valuation.argument",
                "n must be nonnegative",
            )
        base = parse_canonical_integer(self.base)
        if not 2 <= base <= MAX_FACTORIAL_BASE:
            raise PydanticCustomError(
                "number_theory.factorial_valuation.base",
                f"base must be between 2 and {MAX_FACTORIAL_BASE}",
            )
        return _FactorialValuationInput(n=n, base=base)

    @model_validator(mode="after")
    def require_domain(self) -> Self:
        _ = self.admitted
        return self


class FactorialValuationResult(StrictModel):
    n: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    base: CanonicalInteger = Field(max_length=len(str(MAX_FACTORIAL_BASE)))
    valuation: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)

    @model_validator(mode="after")
    def require_domain(self) -> Self:
        if (
            self.n.startswith("-")
            or self.base.startswith("-")
            or self.valuation.startswith("-")
        ):
            raise PydanticCustomError(
                "number_theory.factorial_valuation.result",
                "valuation must be nonnegative",
            )
        return self


class BinomialPrimeValuationRequest(StrictModel):
    """Arguments for the prime valuation of one binomial coefficient."""

    n: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    k: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    prime: CanonicalInteger = Field(max_length=len(str(MAX_BINOMIAL_VALUATION_PRIME)))

    @cached_property
    def admitted(self) -> _BinomialValuationInput:
        n = parse_canonical_integer(self.n)
        k = parse_canonical_integer(self.k)
        prime = parse_canonical_integer(self.prime)
        if n < 0 or not 0 <= k <= n:
            raise PydanticCustomError(
                "number_theory.binomial_valuation.indices",
                "n and k must satisfy 0 <= k <= n",
            )
        if not 2 <= prime <= MAX_BINOMIAL_VALUATION_PRIME:
            raise PydanticCustomError(
                "number_theory.binomial_valuation.prime",
                f"prime must be between 2 and {MAX_BINOMIAL_VALUATION_PRIME}",
            )
        from sympy import isprime

        if not isprime(prime):
            raise PydanticCustomError(
                "number_theory.binomial_valuation.prime",
                "prime must be prime",
            )
        return _BinomialValuationInput(n=n, k=k, prime=prime)

    @model_validator(mode="after")
    def require_domain(self) -> Self:
        _ = self.admitted
        return self


class BinomialPrimeValuationResult(StrictModel):
    n: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    k: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)
    prime: CanonicalInteger = Field(max_length=len(str(MAX_BINOMIAL_VALUATION_PRIME)))
    valuation: CanonicalInteger = Field(max_length=MAX_VALUATION_ARGUMENT_DIGITS)

    @model_validator(mode="after")
    def require_domain(self) -> Self:
        if any(
            value.startswith("-")
            for value in (self.n, self.k, self.prime, self.valuation)
        ):
            raise PydanticCustomError(
                "number_theory.binomial_valuation.result",
                "valuation must be nonnegative",
            )
        return self


@dataclass(frozen=True, slots=True)
class _FactorialValuationInput:
    n: int
    base: int


@dataclass(frozen=True, slots=True)
class _BinomialValuationInput:
    n: int
    k: int
    prime: int


__all__ = [
    "BinomialPrimeValuationRequest",
    "BinomialPrimeValuationResult",
    "FactorialValuationRequest",
    "FactorialValuationResult",
    "FloorSquareRootRequest",
    "FloorSquareRootResult",
    "LegendreSymbolRequest",
    "LegendreSymbolResult",
]
