"""Typed bounded values for initial ideals and Hilbert-function prefixes."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.ideals._models import (
    IdealComputationBudget,
)
from jacobian.math.polynomials.values import RationalPolynomialIdeal

MAX_GRADED_DEGREE = 32
MAX_STANDARD_MONOMIALS = 20_000


class InitialMonomialIdealRequest(StrictModel):
    ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    resource_budget: IdealComputationBudget = Field(default_factory=IdealComputationBudget)


class InitialMonomialIdealResult(StrictModel):
    ideal: RationalPolynomialIdeal
    groebner_basis: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]

    @model_validator(mode="after")
    def require_same_ring(self) -> Self:
        if not (
            self.ideal.variables
            == self.groebner_basis.variables
            == self.initial_ideal.variables
        ):
            raise ValueError("initial-ideal values must share one ordered ring")
        return self


class StandardMonomialsRequest(StrictModel):
    initial_ideal: RationalPolynomialIdeal
    degree: StrictInt = Field(ge=0, le=MAX_GRADED_DEGREE)


class StandardMonomialsResult(StrictModel):
    initial_ideal: RationalPolynomialIdeal
    degree: StrictInt = Field(ge=0, le=MAX_GRADED_DEGREE)
    monomials: tuple[tuple[int, ...], ...]
    count: StrictInt = Field(ge=0, le=MAX_STANDARD_MONOMIALS)

    @model_validator(mode="after")
    def require_count(self) -> Self:
        if self.count != len(self.monomials):
            raise ValueError("standard-monomial count must match the returned list")
        if any(
            len(monomial) != len(self.initial_ideal.variables)
            or sum(monomial) != self.degree
            or any(exponent < 0 for exponent in monomial)
            for monomial in self.monomials
        ):
            raise ValueError("standard monomials must match the ordered ring and degree")
        if self.monomials != tuple(sorted(set(self.monomials), reverse=True)):
            raise ValueError("standard monomials must be unique and canonical")
        return self


class HilbertFunctionRequest(StrictModel):
    ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    max_degree: StrictInt = Field(ge=0, le=MAX_GRADED_DEGREE)
    resource_budget: IdealComputationBudget = Field(default_factory=IdealComputationBudget)


class HilbertFunctionResult(StrictModel):
    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    values: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_source_ring(self) -> Self:
        if self.ideal.variables != self.initial_ideal.variables:
            raise ValueError("Hilbert-function values must share the source ring")
        return self


__all__ = [
    "MAX_GRADED_DEGREE",
    "MAX_STANDARD_MONOMIALS",
    "HilbertFunctionRequest",
    "HilbertFunctionResult",
    "InitialMonomialIdealRequest",
    "InitialMonomialIdealResult",
    "StandardMonomialsRequest",
    "StandardMonomialsResult",
]
