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
    polynomial, so the work grows linearly in coefficient count; the shared
    canonical value alone admits far longer inputs than these kernels establish.
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

    Retains the source polynomial and the original residue for downstream use.
    The producer establishes the Hensel-lift invariant; ordinary result parsing
    checks only the serialized shape.
    """

    polynomial: IntegerPolynomial
    lifted_root: int = Field(ge=0)
    prime: int = Field(ge=2, le=10_000)
    root_mod_p: int = Field(ge=0)
    precision: int = Field(ge=1, le=64)
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
    def _from_kernel(cls, request: HenselRootRequest, lifted_root: int) -> Self:
        return cls.model_construct(
            polynomial=request.polynomial,
            lifted_root=lifted_root,
            prime=request.prime,
            root_mod_p=request.root_mod_p,
            precision=request.precision,
            is_simple_root=True,
        )


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
        request: PAdicRootsRequest,
        roots: tuple[PAdicRootEntry, ...],
        multiple_residues: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            polynomial=request.polynomial,
            roots=roots,
            prime=request.prime,
            precision=request.precision,
            root_count=len(roots),
            multiple_residues=multiple_residues,
        )


def verify_hensel_root_result(result: HenselRootResult) -> None:
    """Verify an independently supplied Hensel-lift claim."""
    request = HenselRootRequest(
        polynomial=result.polynomial,
        prime=result.prime,
        root_mod_p=result.root_mod_p,
        precision=result.precision,
    )
    coefficients = _kernel_coefficients(request.polynomial)
    modulus = request.prime**request.precision
    if (
        result.lifted_root % request.prime != request.root_mod_p
        or _eval_poly_mod(coefficients, result.lifted_root, modulus) != 0
    ):
        raise _validation_error(
            "lifted_root_mismatch", "lifted_root is not the claimed Hensel lift"
        )


def verify_padic_roots_result(result: PAdicRootsResult) -> None:
    """Verify an independently supplied complete simple-root claim."""
    request = PAdicRootsRequest(
        polynomial=result.polynomial, prime=result.prime, precision=result.precision
    )
    coefficients = _kernel_coefficients(request.polynomial)
    roots = tuple(entry.root for entry in result.roots)
    modulus = request.prime**request.precision
    simple_residues = [
        residue
        for residue in range(request.prime)
        if _eval_poly_mod(coefficients, residue, request.prime) == 0
        and _eval_poly_deriv_mod(coefficients, residue, request.prime) != 0
    ]
    multiple = [
        residue
        for residue in range(request.prime)
        if _eval_poly_mod(coefficients, residue, request.prime) == 0
        and _eval_poly_deriv_mod(coefficients, residue, request.prime) == 0
    ]
    if (
        any(_eval_poly_mod(coefficients, root, modulus) != 0 for root in roots)
        or sorted({root % request.prime for root in roots}) != simple_residues
        or sorted(result.multiple_residues) != multiple
    ):
        raise _validation_error(
            "root_profile_mismatch", "roots do not match the source polynomial"
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
