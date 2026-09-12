"""Typed bounded values for initial ideals and Hilbert-function prefixes."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.polynomials.ideals._models import (
    IdealComputationBudget,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialIdeal,
)

MAX_GRADED_DEGREE = 32
MAX_STANDARD_MONOMIALS = 20_000
MAX_HILBERT_SERIES_GENERATORS = 8
MAX_HILBERT_PREFIX = 16


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


class HilbertSeriesRequest(StrictModel):
    ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    prefix_degree: StrictInt = Field(default=0, ge=0, le=MAX_HILBERT_PREFIX)
    resource_budget: IdealComputationBudget = Field(default_factory=IdealComputationBudget)


class HilbertSeriesResult(StrictModel):
    """A bounded Hilbert series with both canonical and graded numerator axes.

    ``reduced_numerator`` is the numerator owned by the canonical rational
    function carrier, whose denominator is monic.  ``h_numerator`` is the
    sign-corrected numerator in the mathematical convention
    ``H(t) = h(t)/(1-t)^d``; the two differ when ``d`` is odd.
    """

    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    ambient_numerator: RationalPolynomial
    ambient_denominator_exponent: StrictInt = Field(ge=0)
    series: RationalFunction
    reduced_numerator: RationalPolynomial
    h_numerator: RationalPolynomial
    denominator_exponent: StrictInt = Field(ge=0)
    prefix: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_source_and_axes(self) -> Self:
        if self.ideal.variables != self.initial_ideal.variables:
            raise ValueError("Hilbert-series values must share the source ring")
        if self.ambient_numerator.variables != ("t",):
            raise ValueError("Hilbert-series numerators use the t axis")
        if (
            self.series.variables != ("t",)
            or self.reduced_numerator.variables != ("t",)
            or self.h_numerator.variables != ("t",)
        ):
            raise ValueError("Hilbert-series values use the t axis")
        if self.series.numerator != self.reduced_numerator.polynomial:
            raise ValueError("reduced numerator must match the rational-series carrier")
        if self.series.denominator.terms:
            denominator_degree = max(self.series.denominator.terms[0].exponents)
            if denominator_degree != self.denominator_exponent:
                raise ValueError("Hilbert-series denominator exponent is inconsistent")
        return self


class HilbertPolynomialResult(StrictModel):
    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    dimension: StrictInt = Field(ge=0)
    polynomial: RationalPolynomial
    stabilization_degree: StrictInt = Field(ge=0)


class HilbertDimensionResult(StrictModel):
    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    dimension: StrictInt = Field(ge=0)


class HilbertMultiplicityResult(StrictModel):
    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    dimension: StrictInt = Field(ge=0)
    multiplicity: ExactInteger = Field(ge=0)


class HVectorResult(StrictModel):
    ideal: RationalPolynomialIdeal
    initial_ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"]
    dimension: StrictInt = Field(ge=0)
    h_vector: tuple[ExactInteger, ...]


__all__ = [
    "MAX_GRADED_DEGREE",
    "MAX_HILBERT_PREFIX",
    "MAX_HILBERT_SERIES_GENERATORS",
    "MAX_STANDARD_MONOMIALS",
    "HVectorResult",
    "HilbertDimensionResult",
    "HilbertFunctionRequest",
    "HilbertFunctionResult",
    "HilbertMultiplicityResult",
    "HilbertPolynomialResult",
    "HilbertSeriesRequest",
    "HilbertSeriesResult",
    "InitialMonomialIdealRequest",
    "InitialMonomialIdealResult",
    "StandardMonomialsRequest",
    "StandardMonomialsResult",
]
