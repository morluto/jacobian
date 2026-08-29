"""Contracts owned by the derived integer kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
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
    """Arguments for the largest exponent ``e`` such that ``base**e`` divides ``n!``.

    Semantic domain (checked once after parse): ``n`` is nonnegative and
    ``base`` lies in ``[2, 1000000]``.
    """

    n: CanonicalInteger = Field(
        max_length=MAX_VALUATION_ARGUMENT_DIGITS,
        description=(
            f"Nonnegative integer n with at most {MAX_VALUATION_ARGUMENT_DIGITS} "
            "decimal digits."
        ),
    )
    base: CanonicalInteger = Field(
        max_length=len(str(MAX_FACTORIAL_BASE)),
        description=f"Integer base in [2, {MAX_FACTORIAL_BASE}].",
    )


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
    """Arguments for the prime valuation of one binomial coefficient.

    Semantic domain (checked once after parse): ``0 <= k <= n`` and ``prime`` is
    an ordinary prime at most ``2**64 - 1`` (SymPy's deterministic primality
    ceiling).
    """

    n: CanonicalInteger = Field(
        max_length=MAX_VALUATION_ARGUMENT_DIGITS,
        description=(
            f"Nonnegative upper index n with at most {MAX_VALUATION_ARGUMENT_DIGITS} "
            "decimal digits; must satisfy 0 <= k <= n."
        ),
    )
    k: CanonicalInteger = Field(
        max_length=MAX_VALUATION_ARGUMENT_DIGITS,
        description=(
            f"Nonnegative lower index k with at most {MAX_VALUATION_ARGUMENT_DIGITS} "
            "decimal digits; must satisfy 0 <= k <= n."
        ),
    )
    prime: CanonicalInteger = Field(
        max_length=len(str(MAX_BINOMIAL_VALUATION_PRIME)),
        description=(
            f"Ordinary prime p in [2, {MAX_BINOMIAL_VALUATION_PRIME}] "
            "(deterministic SymPy primality range)."
        ),
    )


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


def admit_factorial_valuation(n: int, base: int) -> _FactorialValuationInput:
    """Admit one factorial-valuation request after structural parsing."""

    if type(n) is not int or n < 0 or n >= 10**MAX_VALUATION_ARGUMENT_DIGITS:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.factorial_valuation.argument_bound",
            message=(
                "n must be a nonnegative integer with at most "
                f"{MAX_VALUATION_ARGUMENT_DIGITS} decimal digits"
            ),
        )
    if type(base) is not int or not 2 <= base <= MAX_FACTORIAL_BASE:
        raise OperationDomainValidationError(
            location=("base",),
            code="number_theory.factorial_valuation.base_bound",
            message=f"base must be between 2 and {MAX_FACTORIAL_BASE}",
        )
    return _FactorialValuationInput(n=n, base=base)


def admit_binomial_prime_valuation(
    n: int, k: int, prime: int
) -> _BinomialValuationInput:
    """Admit one binomial prime-valuation request after structural parsing."""

    if (
        type(n) is not int
        or type(k) is not int
        or n < 0
        or not 0 <= k <= n
        or n >= 10**MAX_VALUATION_ARGUMENT_DIGITS
    ):
        raise OperationDomainValidationError(
            location=("n", "k"),
            code="number_theory.binomial_valuation.indices",
            message=(
                "n and k must satisfy 0 <= k <= n and use at most "
                f"{MAX_VALUATION_ARGUMENT_DIGITS} decimal digits"
            ),
        )
    if type(prime) is not int or not 2 <= prime <= MAX_BINOMIAL_VALUATION_PRIME:
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.binomial_valuation.prime_required",
            message=(
                "prime must be a prime number between 2 and "
                f"{MAX_BINOMIAL_VALUATION_PRIME}"
            ),
        )
    from sympy import isprime

    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.binomial_valuation.prime_required",
            message="prime must be prime",
        )
    return _BinomialValuationInput(n=n, k=k, prime=prime)


__all__ = [
    "BinomialPrimeValuationRequest",
    "BinomialPrimeValuationResult",
    "FactorialValuationRequest",
    "FactorialValuationResult",
    "FloorSquareRootRequest",
    "FloorSquareRootResult",
    "LegendreSymbolRequest",
    "LegendreSymbolResult",
    "admit_binomial_prime_valuation",
    "admit_factorial_valuation",
]
