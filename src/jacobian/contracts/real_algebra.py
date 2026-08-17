"""Typed wire contracts for exact real algebra operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import (
    CanonicalRational,
    require_bounded_rational,
)

MAX_POLYNOMIAL_DEGREE = 32
MAX_POLYNOMIAL_TERMS = 33
MAX_COEFFICIENT_DIGITS = 256


class PolynomialTerm(ContractModel):
    """One term: coefficient times x^exponent."""

    coefficient: CanonicalRational
    exponent: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)


class UnivariatePolynomial(ContractModel):
    """A univariate polynomial over QQ as sparse nonzero terms."""

    terms: tuple[PolynomialTerm, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_TERMS
    )

    @model_validator(mode="after")
    def require_unique_exponents(self) -> Self:
        exponents = [t.exponent for t in self.terms]
        if len(set(exponents)) != len(exponents):
            raise ValueError("polynomial exponents must be unique")
        if any(t.coefficient.as_fraction() == 0 for t in self.terms):
            raise ValueError("zero polynomial terms must be omitted")
        for term in self.terms:
            require_bounded_rational(
                term.coefficient,
                max_digits=MAX_COEFFICIENT_DIGITS,
                label="polynomial coefficient",
            )
        return self


class SturmChainRequest(ContractModel):
    """Compute the Sturm chain of a univariate polynomial."""

    polynomial: UnivariatePolynomial

    @model_validator(mode="after")
    def require_nonconstant_polynomial(self) -> Self:
        if max(t.exponent for t in self.polynomial.terms) < 1:
            raise ValueError("Sturm chain requires a non-constant polynomial")
        return self


class RootCountRequest(ContractModel):
    """Count real roots of a polynomial in an interval [a, b]."""

    polynomial: UnivariatePolynomial
    lower: CanonicalRational
    upper: CanonicalRational

    @model_validator(mode="after")
    def require_lower_leq_upper(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise ValueError("lower bound must not exceed upper bound")
        return self


class SturmChainResult(ContractModel):
    """The Sturm chain as a list of polynomials."""

    chain: tuple[UnivariatePolynomial, ...] = Field(min_length=1)
    degree: int = Field(ge=1, le=MAX_POLYNOMIAL_DEGREE)
    method: Literal["SYMPY_STURM"] = "SYMPY_STURM"


class RootCountResult(ContractModel):
    """Count of real roots in an interval."""

    root_count: int = Field(ge=0)
    lower: CanonicalRational
    upper: CanonicalRational
    method: Literal["STURM_THEOREM"] = "STURM_THEOREM"


__all__ = [
    "PolynomialTerm",
    "RootCountRequest",
    "RootCountResult",
    "SturmChainRequest",
    "SturmChainResult",
    "UnivariatePolynomial",
]
