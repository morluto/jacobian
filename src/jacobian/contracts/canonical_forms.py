"""Typed wire contracts for exact canonical-form operations over QQ."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import (
    CanonicalRational,
)
from jacobian.contracts.matrices import RationalMatrix, require_matrix_scalar_digits

MAX_CANONICAL_FORM_DIMENSION = 16
MAX_CANONICAL_FORM_SCALAR_DIGITS = 256


class SquareMatrixRequest(ContractModel):
    """One square rational matrix bounded for canonical-form computation."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_bounded_square(self) -> Self:
        n = len(self.matrix.entries)
        if n != len(self.matrix.entries[0]):
            raise ValueError("canonical-form operations require a square matrix")
        if n > MAX_CANONICAL_FORM_DIMENSION:
            raise ValueError(
                "canonical-form operations are bounded to 16 x 16 matrices"
            )
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_CANONICAL_FORM_SCALAR_DIGITS,
            label="canonical-form matrix",
        )
        return self


class MonicPolynomial(ContractModel):
    """One monic univariate polynomial over QQ, as increasing-degree coefficients."""

    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_monic(self) -> Self:
        if self.coefficients[-1].as_fraction() != 1:
            raise ValueError("polynomial must be monic (leading coefficient = 1)")
        if len(self.coefficients) > MAX_CANONICAL_FORM_DIMENSION + 1:
            raise ValueError("polynomial degree exceeds the canonical-form bound")
        return self

    def as_sympy(self) -> object:
        from sympy import Symbol

        x = Symbol("x")
        return sum(
            self.coefficients[i].as_fraction() * x**i
            for i in range(len(self.coefficients))
        )


class MinimalPolynomialResult(ContractModel):
    """Exact minimal polynomial of a square rational matrix."""

    minimal_polynomial: MonicPolynomial
    characteristic_polynomial: MonicPolynomial
    degree: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["KRYLOV_NULLSPACE"] = "KRYLOV_NULLSPACE"


class InvariantFactorEntry(ContractModel):
    """One monic invariant factor from the rational canonical form."""

    factor: MonicPolynomial
    block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)


class RationalCanonicalFormResult(ContractModel):
    """Exact rational (Frobenius) canonical form of a square rational matrix."""

    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    total_block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["SMITH_NORMAL_FORM"] = "SMITH_NORMAL_FORM"


class PrimaryDecompositionResult(ContractModel):
    """Primary decomposition of the minimal polynomial into irreducible-power components."""

    components: tuple[MonicPolynomial, ...] = Field(min_length=1)
    minimal_polynomial: MonicPolynomial
    method: Literal["FACTOR_LCM"] = "FACTOR_LCM"


__all__ = [
    "InvariantFactorEntry",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SquareMatrixRequest",
]
