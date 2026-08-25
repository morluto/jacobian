"""Typed wire contracts for p-adic number theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials._models import IntegerPolynomial

_MAX_PADIC_COEFFICIENTS = 64


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _require_padic_budget(polynomial: IntegerPolynomial) -> None:
    """Bound one admitted polynomial for the p-adic kernels.

    Root finding evaluates every residue mod ``p`` against the source
    polynomial and result replay repeats that sweep, so the work grows
    linearly in coefficient count; the shared canonical value alone
    admits far longer inputs than these kernels establish.
    """
    if len(polynomial.coefficients) > _MAX_PADIC_COEFFICIENTS:
        raise _validation_error(
            "padic_arithmetic.polynomial_budget",
            "p-adic polynomial exceeds the 64-coefficient operation budget",
        )


def _kernel_coefficients(polynomial: IntegerPolynomial) -> tuple[int, ...]:
    """Convert the canonical descending ``ZZ[x]`` value to ascending ints.

    The canonical integer-polynomial value stores decimal-string
    coefficients highest degree first; every kernel and validator below
    indexes coefficient ``i`` as the degree-``i`` term, so the explicit
    order conversion happens exactly here and nowhere else.
    """
    return tuple(
        parse_canonical_integer(coefficient)
        for coefficient in reversed(polynomial.coefficients)
    )


MAX_PRIME = 10_000


def _require_prime(value: int) -> None:
    """Reject composite moduli: the field is documented and used as GF(p)."""
    if (
        value < 2
        or value > MAX_PRIME
        or any(value % divisor == 0 for divisor in range(2, int(value**0.5) + 1))
    ):
        raise _validation_error(
            "padic_arithmetic.prime_not_prime", "prime must be a prime modulus"
        )


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
    def require_operation_budget(self) -> Self:
        _require_padic_budget(self.polynomial)
        return self

    @model_validator(mode="after")
    def require_valid_root(self) -> Self:
        _require_prime(self.prime)
        if self.root_mod_p >= self.prime:
            raise _validation_error(
                "padic_arithmetic.root_out_of_range", "root_mod_p must be in 0..p-1"
            )
        # The contract lifts a SIMPLE root: f(r) = 0 and f'(r) != 0 mod p are
        # mathematical preconditions, so validate them at this boundary.
        coeffs = _kernel_coefficients(self.polynomial)
        if _poly_eval_mod_p(coeffs, self.root_mod_p, self.prime) != 0:
            raise _validation_error(
                "padic_arithmetic.root_not_root",
                "root_mod_p must satisfy f(root_mod_p) = 0 mod p",
            )
        if _poly_deriv_mod_p(coeffs, self.root_mod_p, self.prime) % self.prime == 0:
            raise _validation_error(
                "padic_arithmetic.root_not_simple",
                "Hensel lifting requires a simple root: "
                "f'(root_mod_p) must be nonzero mod p",
            )
        return self


class HenselRootResult(StrictModel):
    """A p-adic root approximation lifted via Hensel's lemma.

    Retains the source polynomial and the original residue so the lift can
    be replayed: the residue is a simple root of f mod p, the lifted root
    reduces to it mod p, and it vanishes modulo p^precision.
    """

    polynomial: IntegerPolynomial
    lifted_root: int = Field(ge=0)
    prime: int = Field(ge=2, le=10_000)
    root_mod_p: int = Field(ge=0)
    precision: int = Field(ge=1, le=64)
    is_simple_root: bool

    @model_validator(mode="after")
    def require_valid_lifted_root(self) -> Self:
        p = self.prime
        modulus = p**self.precision
        _require_prime(p)
        _require_padic_budget(self.polynomial)
        if self.root_mod_p >= p:
            raise _validation_error(
                "padic_arithmetic.root_out_of_range", "root_mod_p must be in 0..p-1"
            )
        if self.lifted_root >= modulus:
            raise _validation_error(
                "padic_arithmetic.lifted_root_out_of_range",
                "lifted_root must be in 0..p^k - 1",
            )
        coeffs = _kernel_coefficients(self.polynomial)
        # Replay the defining invariants against the retained source data so
        # a serialized result cannot detach its lift from its polynomial.
        if _poly_eval_mod_p(coeffs, self.root_mod_p, p) != 0:
            raise _validation_error(
                "padic_arithmetic.root_not_root",
                "root_mod_p must satisfy f(root_mod_p) = 0 mod p",
            )
        if _poly_deriv_mod_p(coeffs, self.root_mod_p, p) % p == 0:
            raise _validation_error(
                "padic_arithmetic.root_not_simple",
                "the lifted residue must be simple: "
                "f'(root_mod_p) must be nonzero mod p",
            )
        if self.lifted_root % p != self.root_mod_p:
            raise _validation_error(
                "padic_arithmetic.lifted_root_wrong_residue",
                "lifted_root must reduce to root_mod_p modulo the prime",
            )
        if _poly_eval_mod_p(coeffs, self.lifted_root, modulus) != 0:
            raise _validation_error(
                "padic_arithmetic.lifted_root_not_root",
                "lifted_root must satisfy f(lifted_root) = 0 modulo p^precision",
            )
        if not self.is_simple_root:
            raise _validation_error(
                "padic_arithmetic.simple_flag_invalid",
                "only simple roots are lifted, so the flag must hold",
            )
        return self


class HenselFactorLiftRequest(StrictModel):
    """Lift a coprime factorization f ≡ g*h (mod p) to f ≡ g*h (mod p^k)."""

    polynomial: IntegerPolynomial
    factor_g: IntegerPolynomial
    factor_h: IntegerPolynomial
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_padic_budget(self.polynomial)
        _require_padic_budget(self.factor_g)
        _require_padic_budget(self.factor_h)
        return self

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
    def require_operation_budget(self) -> Self:
        _require_padic_budget(self.polynomial)
        return self

    @model_validator(mode="after")
    def require_prime_modulus(self) -> Self:
        _require_prime(self.prime)
        return self


class PAdicRootEntry(StrictModel):
    """One exact root of f mod p^k lifted from a simple root mod p."""

    root: int = Field(ge=0)
    root_type: Literal["SIMPLE"] = "SIMPLE"


def _eval_poly_mod(coefficients: tuple[int, ...], x: int, modulus: int) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x + coefficient) % modulus
    return result


def _eval_poly_deriv_mod(coefficients: tuple[int, ...], x: int, modulus: int) -> int:
    result = 0
    for index in range(len(coefficients) - 1, 0, -1):
        result = (result * x + index * coefficients[index]) % modulus
    return result


class PAdicRootsResult(StrictModel):
    """Every simple root of f(x) mod p^k plus unlifted multiple residues.

    ``roots`` lists the unique Hensel lift of each simple root mod p, so each
    entry is an exact root modulo p^precision. ``multiple_residues`` lists
    residues r mod p with f(r) = f'(r) = 0 (mod p); their lifts are not
    enumerated because the mod-p^k solution set can grow unboundedly.
    """

    polynomial: IntegerPolynomial
    roots: tuple[PAdicRootEntry, ...]
    prime: int = Field(ge=2, le=10_000)
    precision: int = Field(ge=1, le=64)
    root_count: int = Field(ge=0)
    multiple_residues: tuple[int, ...] = ()

    @model_validator(mode="after")
    def require_consistent_count(self) -> Self:
        _require_padic_budget(self.polynomial)
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
        return self

    @model_validator(mode="after")
    def require_source_bound_roots(self) -> Self:
        p = self.prime
        k = self.precision
        # The advertised semantics are p-adic: validate the prime modulus
        # before replaying the root set against it.
        _require_prime(p)
        modulus = p**k
        coefficients = _kernel_coefficients(self.polynomial)
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
        # Replay every classification against the retained polynomial: each
        # listed root must be an exact vanishing simple root, and the simple
        # versus multiple split must be complete over all residues mod p.
        for root in roots:
            if _eval_poly_mod(coefficients, root, modulus) != 0:
                raise _validation_error(
                    "padic_arithmetic.root_not_root",
                    "each root must satisfy f(root) = 0 modulo p^precision",
                )
            if _eval_poly_deriv_mod(coefficients, root, p) == 0:
                raise _validation_error(
                    "padic_arithmetic.root_not_simple",
                    "each lifted root must be simple modulo p",
                )
        simple_residues = [
            residue
            for residue in range(p)
            if _eval_poly_mod(coefficients, residue, p) == 0
            and _eval_poly_deriv_mod(coefficients, residue, p) != 0
        ]
        multiple = [
            residue
            for residue in range(p)
            if _eval_poly_mod(coefficients, residue, p) == 0
            and _eval_poly_deriv_mod(coefficients, residue, p) == 0
        ]
        if sorted({root % p for root in roots}) != simple_residues:
            raise _validation_error(
                "padic_arithmetic.simple_residues_mismatch",
                "roots must cover exactly the simple residues of f modulo p",
            )
        if len(roots) != len(simple_residues):
            raise _validation_error(
                "padic_arithmetic.simple_root_count_mismatch",
                "each simple residue lifts to exactly one root",
            )
        if sorted(self.multiple_residues) != multiple:
            raise _validation_error(
                "padic_arithmetic.multiple_residues_mismatch",
                "multiple_residues must list every repeated-factor residue of f mod p",
            )
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
