"""Typed wire contracts for p-adic number theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials._models import IntegerPolynomial


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _kernel_coefficients(polynomial: IntegerPolynomial) -> tuple[int, ...]:
    """Convert the canonical descending ``ZZ[x]`` value to ascending ints.

    The canonical integer-polynomial value stores decimal-string
    coefficients highest degree first; every kernel indexes
    indexes coefficient ``i`` as the degree-``i`` term, so the explicit
    order conversion happens exactly here and nowhere else.
    """
    return tuple(
        parse_canonical_integer(coefficient)
        for coefficient in reversed(polynomial.coefficients)
    )


MAX_PRIME = 10_000
MAX_PRECISION = 64


class HenselRootRequest(StrictModel):
    """Lift a simple root of f(x) mod p to a root mod p^k.

    Given f in ZZ[x], a prime p, and a root r with f(r) ≡ 0 (mod p)
    and f'(r) not ≡ 0 (mod p), lift r to a root mod p^k.
    """

    polynomial: IntegerPolynomial
    prime: int = Field(ge=2, le=MAX_PRIME)
    root_mod_p: int = Field(ge=0)
    precision: int = Field(ge=1, le=MAX_PRECISION)


class HenselRootResult(StrictModel):
    """A p-adic root approximation lifted via Hensel's lemma.

    Retains the source polynomial and the original residue for downstream use.
    The producer establishes the Hensel-lift invariant; ordinary result parsing
    checks only the serialized shape.
    """

    polynomial: IntegerPolynomial
    lifted_root: int = Field(ge=0)
    prime: int = Field(ge=2, le=MAX_PRIME)
    root_mod_p: int = Field(ge=0)
    precision: int = Field(ge=1, le=MAX_PRECISION)
    is_simple_root: bool

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        if self.root_mod_p >= self.prime:
            raise _validation_error(
                "padic_arithmetic.root_out_of_range", "root_mod_p must be in 0..p-1"
            )
        if self.lifted_root >= self.prime**self.precision:
            raise _validation_error(
                "padic_arithmetic.lifted_root_out_of_range",
                "lifted_root must be in 0..p^k - 1",
            )
        if not self.is_simple_root:
            raise _validation_error(
                "padic_arithmetic.simple_flag_invalid",
                "only simple roots are lifted, so the flag must hold",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        polynomial: IntegerPolynomial,
        prime: int,
        root_mod_p: int,
        precision: int,
        lifted_root: int,
    ) -> Self:
        return cls.model_construct(
            polynomial=polynomial,
            lifted_root=lifted_root,
            prime=prime,
            root_mod_p=root_mod_p,
            precision=precision,
            is_simple_root=True,
        )


class HenselFactorLiftRequest(StrictModel):
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k)."""

    polynomial: IntegerPolynomial
    factor_g: IntegerPolynomial
    factor_h: IntegerPolynomial
    prime: int = Field(ge=2, le=MAX_PRIME)
    precision: int = Field(ge=1, le=MAX_PRECISION)


class HenselFactorLiftResult(StrictModel):
    """Lifted coprime factors mod p^k."""

    lifted_g: IntegerPolynomial
    lifted_h: IntegerPolynomial
    prime: int = Field(ge=2, le=MAX_PRIME)
    precision: int = Field(ge=1, le=MAX_PRECISION)


class PAdicRootsRequest(StrictModel):
    """Find every simple root of f(x) mod p^k via Hensel lifting.

    Roots mod p with nonzero derivative lift uniquely to roots mod p^k.
    Residues r mod p with f(r) = f'(r) = 0 (mod p) are reported in
    ``multiple_residues`` without lifting: their mod-p^k root sets can grow
    unboundedly (e.g. x^2 has five roots mod 25), so they are not lifted.
    """

    polynomial: IntegerPolynomial
    prime: int = Field(ge=2, le=MAX_PRIME)
    precision: int = Field(ge=1, le=MAX_PRECISION)


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

    polynomial: IntegerPolynomial
    roots: tuple[PAdicRootEntry, ...]
    prime: int = Field(ge=2, le=MAX_PRIME)
    precision: int = Field(ge=1, le=MAX_PRECISION)
    root_count: int = Field(ge=0)
    multiple_residues: tuple[int, ...] = ()

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        if self.root_count != len(self.roots):
            raise _validation_error(
                "padic_arithmetic.root_count_mismatch",
                "root_count must match the number of roots",
            )
        if len(set(self.multiple_residues)) != len(self.multiple_residues):
            raise _validation_error(
                "padic_arithmetic.multiple_residues_not_distinct",
                "multiple residues must be distinct",
            )
        if any(r >= self.prime for r in self.multiple_residues):
            raise _validation_error(
                "padic_arithmetic.multiple_residue_out_of_range",
                "multiple residues must lie in 0..p-1",
            )
        modulus = self.prime**self.precision
        roots = tuple(entry.root for entry in self.roots)
        if any(root >= modulus for root in roots):
            raise _validation_error(
                "padic_arithmetic.root_out_of_range", "roots must lie in 0..p^k - 1"
            )
        if len(set(roots)) != len(roots):
            raise _validation_error(
                "padic_arithmetic.roots_not_distinct",
                "roots must be distinct modulo p^k",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        polynomial: IntegerPolynomial,
        prime: int,
        precision: int,
        roots: tuple[PAdicRootEntry, ...],
        multiple_residues: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            polynomial=polynomial,
            roots=roots,
            prime=prime,
            precision=precision,
            root_count=len(roots),
            multiple_residues=multiple_residues,
        )


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
