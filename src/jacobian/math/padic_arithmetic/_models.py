"""Typed wire contracts for p-adic number theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

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


MAX_PRIME = 10_000


def _require_prime(value: int) -> None:
    """Reject composite moduli: the field is documented and used as GF(p)."""
    if value < 2 or value > MAX_PRIME or any(
        value % divisor == 0 for divisor in range(2, int(value**0.5) + 1)
    ):
        raise ValueError("prime must be a prime modulus")


def _poly_eval_mod_p(coeffs: tuple[int, ...], x: int, p: int) -> int:
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % p
    return result


def _poly_deriv_mod_p(coeffs: tuple[int, ...], x: int, p: int) -> int:
    if len(coeffs) <= 1:
        return 0
    deriv = tuple(i * coeffs[i] for i in range(1, len(coeffs)))
    return _poly_eval_mod_p(deriv, x, p)


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
        _require_prime(self.prime)
        if self.root_mod_p >= self.prime:
            raise ValueError("root_mod_p must be in 0..p-1")
        # The contract lifts a SIMPLE root: f(r) = 0 and f'(r) != 0 mod p are
        # mathematical preconditions, so validate them at this boundary.
        coeffs = self.polynomial.coefficients
        if _poly_eval_mod_p(coeffs, self.root_mod_p, self.prime) != 0:
            raise ValueError("root_mod_p must satisfy f(root_mod_p) = 0 mod p")
        if _poly_deriv_mod_p(coeffs, self.root_mod_p, self.prime) % self.prime == 0:
            raise ValueError(
                "Hensel lifting requires a simple root: "
                "f'(root_mod_p) must be nonzero mod p"
            )
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

    @model_validator(mode="after")
    def require_prime_modulus(self) -> Self:
        _require_prime(self.prime)
        return self


class HenselFactorLiftResult(StrictModel):
    """Lifted coprime factors mod p^k."""

    lifted_g: IntegerPolynomial
    lifted_h: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)


class PAdicRootsRequest(StrictModel):
    """Find every simple root of f(x) mod p^k via Hensel lifting.

    Roots mod p with nonzero derivative lift uniquely to roots mod p^k.
    Residues r mod p with f(r) = f'(r) = 0 (mod p) are reported in
    ``multiple_residues`` without lifting: their mod-p^k root sets can grow
    unboundedly (e.g. x^2 has five roots mod 25), so they are not lifted.
    """

    polynomial: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def require_prime_modulus(self) -> Self:
        _require_prime(self.prime)
        return self


class PAdicRootEntry(StrictModel):
    """One exact root of f mod p^k lifted from a simple root mod p."""

    root: int = Field(ge=0)
    root_type: Literal["SIMPLE"] = "SIMPLE"


class PAdicRootsResult(StrictModel):
    """Every simple root of f(x) mod p^k plus unlifted multiple residues.

    ``roots`` lists the unique Hensel lift of each simple root mod p, so each
    entry is an exact root modulo p^precision. ``multiple_residues`` lists
    residues r mod p with f(r) = f'(r) = 0 (mod p); their lifts are not
    enumerated because the mod-p^k solution set can grow unboundedly.
    """

    roots: tuple[PAdicRootEntry, ...]
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)
    root_count: int = Field(ge=0)
    multiple_residues: tuple[int, ...] = ()

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        if self.root_count != len(self.roots):
            raise ValueError("root_count must match the number of roots")
        if len(set(self.multiple_residues)) != len(self.multiple_residues):
            raise ValueError("multiple residues must be distinct")
        if any(r >= self.prime for r in self.multiple_residues):
            raise ValueError("multiple residues must lie in 0..p-1")
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
