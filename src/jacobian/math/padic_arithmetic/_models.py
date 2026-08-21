"""Typed wire contracts for p-adic number theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel


class IntegerPolynomial(StrictModel):
    """A univariate integer polynomial a_0 + a_1*x + ... + a_n*x^n."""

    coefficients: tuple[int, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_valid_coefficients(self) -> Self:
        if any(
            type(c) is not int or c < -(2**31) or c >= 2**31
            for c in self.coefficients
        ):
            raise ValueError("coefficients must be bounded integers")
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1


class HenselRootRequest(StrictModel):
    """Lift a simple root of f(x) mod p to a root mod p^k.

    Given f in ZZ[x], a prime p, and a root r with f(r) ≡ 0 (mod p)
    and f'(r) not ≡ 0 (mod p), lift r to a root mod p^k.
    """

    polynomial: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    root_mod_p: int = Field(ge=0)
    precision: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def require_valid_root(self) -> Self:
        if self.root_mod_p >= self.prime:
            raise ValueError("root_mod_p must be in 0..p-1")
        return self


class HenselRootResult(StrictModel):
    """A p-adic root approximation lifted via Hensel's lemma."""

    lifted_root: int = Field(ge=0)
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)
    is_simple_root: bool

    @model_validator(mode="after")
    def require_valid_lifted_root(self) -> Self:
        modulus = self.prime**self.precision
        if self.lifted_root < 0 or self.lifted_root >= modulus:
            raise ValueError("lifted_root must be in 0..p^k - 1")
        return self


class HenselFactorLiftRequest(StrictModel):
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k)."""

    polynomial: IntegerPolynomial
    factor_g: IntegerPolynomial
    factor_h: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)


class HenselFactorLiftResult(StrictModel):
    """Lifted coprime factors mod p^k."""

    lifted_g: IntegerPolynomial
    lifted_h: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)


class PAdicRootsRequest(StrictModel):
    """Find all roots of f(x) mod p^k via Hensel lifting."""

    polynomial: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)


class PAdicRootEntry(StrictModel):
    """One p-adic root with its lift type."""

    root: int = Field(ge=0)
    root_type: Literal["SIMPLE", "MULTIPLE"] = "SIMPLE"


class PAdicRootsResult(StrictModel):
    """All roots of f(x) mod p^k."""

    roots: tuple[PAdicRootEntry, ...]
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)
    root_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if self.root_count != len(self.roots):
            raise ValueError("root_count must match the number of roots")
        return self


__all__ = [
    "HenselFactorLiftRequest",
    "HenselFactorLiftResult",
    "HenselRootRequest",
    "HenselRootResult",
    "IntegerPolynomial",
    "PAdicRootEntry",
    "PAdicRootsRequest",
    "PAdicRootsResult",
]
