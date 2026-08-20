"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_VARIABLES = 8
_MAX_MAP_OUTPUTS = 20
_MAX_TERMS = 256
_MAX_EXPONENT = 64
_MAX_COEFFICIENT_DIGITS = 128
_MAX_COMPOSITION_DEGREE = 128


def _require_map_polynomial(polynomial: RationalPolynomial, *, label: str) -> None:
    if len(polynomial.variables) > _MAX_VARIABLES:
        raise ValueError(f"{label} exceeds the {_MAX_VARIABLES}-variable budget")
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_TERMS,
        maximum_exponent=_MAX_EXPONENT,
        maximum_coefficient_digits=_MAX_COEFFICIENT_DIGITS,
        label=label,
    )
    if any(sum(term.exponents) > _MAX_EXPONENT for term in polynomial.polynomial.terms):
        raise ValueError(f"{label} exceeds total degree {_MAX_EXPONENT}")


class VariablePoint(StrictModel):
    """One rational point on an explicitly ordered polynomial axis."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=_MAX_VARIABLES
    )
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=_MAX_VARIABLES
    )

    @model_validator(mode="after")
    def require_matching_axis(self) -> Self:
        if len(self.variables) != len(self.values):
            raise ValueError("point variables and values must have the same length")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("point variables must be unique")
        return self


class EvalRequest(StrictModel):
    """Evaluate one canonical rational polynomial at a complete rational point."""

    polynomial: RationalPolynomial
    point: VariablePoint

    @model_validator(mode="after")
    def require_complete_ordered_point(self) -> Self:
        _require_map_polynomial(self.polynomial, label="evaluation polynomial")
        if self.point.variables != self.polynomial.variables:
            raise ValueError(
                "evaluation point must use the polynomial's complete ordered axis"
            )
        value = Fraction(0)
        coordinates = tuple(item.as_fraction() for item in self.point.values)
        for term in self.polynomial.polynomial.terms:
            monomial = term.coefficient.as_fraction()
            for coordinate, exponent in zip(coordinates, term.exponents, strict=True):
                monomial *= coordinate**exponent
            value += monomial
        CanonicalRational.from_fraction(value)
        return self


class EvalResult(StrictModel):
    """The exact rational value at the requested point."""

    value: CanonicalRational


class JacobianRequest(StrictModel):
    """Compute the Jacobian of a canonical polynomial map."""

    input_variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=_MAX_VARIABLES
    )
    output_polynomials: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=_MAX_MAP_OUTPUTS
    )

    @model_validator(mode="after")
    def require_one_map_ring(self) -> Self:
        if len(set(self.input_variables)) != len(self.input_variables):
            raise ValueError("input variables must be unique")
        for polynomial in self.output_polynomials:
            _require_map_polynomial(polynomial, label="map output polynomial")
            if polynomial.variables != self.input_variables:
                raise ValueError(
                    "every map output must use the complete ordered input axis"
                )
        return self


class JacobianResult(StrictModel):
    """The row-major Jacobian matrix over the source polynomial ring."""

    n_inputs: int = Field(ge=1, le=_MAX_VARIABLES)
    n_outputs: int = Field(ge=1, le=_MAX_MAP_OUTPUTS)
    entries: tuple[RationalPolynomial, ...] = Field(max_length=160)

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if len(self.entries) != self.n_inputs * self.n_outputs:
            raise ValueError("Jacobian entry count must match its matrix dimensions")
        if self.entries:
            variables = self.entries[0].variables
            if any(entry.variables != variables for entry in self.entries):
                raise ValueError("Jacobian entries must use one ordered ring")
        return self


class CompositionRequest(StrictModel):
    """Compose two bounded univariate rational polynomials."""

    outer: RationalPolynomial
    inner: RationalPolynomial
    inner_variable: PolynomialVariable
    outer_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_univariate_bounded_composition(self) -> Self:
        _require_map_polynomial(self.outer, label="outer polynomial")
        _require_map_polynomial(self.inner, label="inner polynomial")
        if self.outer.variables != (self.outer_variable,):
            raise ValueError("outer polynomial must use exactly outer_variable")
        if self.inner.variables != (self.inner_variable,):
            raise ValueError("inner polynomial must use exactly inner_variable")
        outer_degree = max(
            (term.exponents[0] for term in self.outer.polynomial.terms), default=0
        )
        inner_degree = max(
            (term.exponents[0] for term in self.inner.polynomial.terms), default=0
        )
        if outer_degree * inner_degree > _MAX_COMPOSITION_DEGREE:
            raise ValueError(f"composition exceeds degree {_MAX_COMPOSITION_DEGREE}")
        return self


class CompositionResult(StrictModel):
    """The canonical polynomial obtained by substitution."""

    polynomial: RationalPolynomial


__all__ = [
    "CompositionRequest",
    "CompositionResult",
    "EvalRequest",
    "EvalResult",
    "JacobianRequest",
    "JacobianResult",
    "VariablePoint",
]
