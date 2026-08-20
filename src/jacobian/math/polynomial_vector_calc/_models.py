"""Typed wire contracts for polynomial vector calculus operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_VARS = 8
MAX_POLYS = 8
_MAX_TERMS = 256
_MAX_EXPONENT = 64
_MAX_COEFFICIENT_DIGITS = 128


def _require_field_polynomial(
    polynomial: RationalPolynomial,
    *,
    label: str,
) -> None:
    if len(polynomial.variables) > MAX_VARS:
        raise ValueError(f"{label} exceeds the {MAX_VARS}-variable budget")
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_TERMS,
        maximum_exponent=_MAX_EXPONENT,
        maximum_coefficient_digits=_MAX_COEFFICIENT_DIGITS,
        label=label,
    )
    if any(sum(term.exponents) > _MAX_EXPONENT for term in polynomial.polynomial.terms):
        raise ValueError(f"{label} exceeds total degree {_MAX_EXPONENT}")


class ScalarFieldRequest(StrictModel):
    """One bounded canonical multivariate polynomial scalar field."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_bounded_field(self) -> Self:
        _require_field_polynomial(self.polynomial, label="scalar field")
        if (
            len(self.polynomial.polynomial.terms) * len(self.polynomial.variables)
            > _MAX_TERMS
        ):
            raise ValueError("scalar-field derivatives exceed the result-term budget")
        return self


class VectorFieldRequest(StrictModel):
    """A polynomial vector field with one component per ordered variable."""

    components: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_POLYS
    )

    @model_validator(mode="after")
    def require_one_vector_field_ring(self) -> Self:
        variables = self.components[0].variables
        if len(self.components) != len(variables):
            raise ValueError("vector field must have one component per variable")
        for component in self.components:
            _require_field_polynomial(component, label="vector-field component")
            if component.variables != variables:
                raise ValueError("vector-field components must use one ordered ring")
        if sum(len(item.polynomial.terms) for item in self.components) > _MAX_TERMS:
            raise ValueError("vector-field derivatives exceed the result-term budget")
        return self


class CurlRequest(VectorFieldRequest):
    """A three-dimensional polynomial vector field."""

    @model_validator(mode="after")
    def require_three_dimensions(self) -> Self:
        if len(self.components[0].variables) != 3:
            raise ValueError("curl requires exactly three variables and components")
        return self


class DirectionalDerivativeRequest(StrictModel):
    """Directional derivative along one exact constant vector."""

    polynomial: RationalPolynomial
    direction: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=MAX_VARS)

    @model_validator(mode="after")
    def require_matching_bounded_direction(self) -> Self:
        _require_field_polynomial(self.polynomial, label="scalar field")
        if len(self.direction) != len(self.polynomial.variables):
            raise ValueError("direction vector length must match the polynomial axis")
        for coordinate in self.direction:
            require_bounded_rational(
                coordinate,
                max_digits=_MAX_COEFFICIENT_DIGITS,
                label="direction coordinate",
            )
        if (
            len(self.polynomial.polynomial.terms) * len(self.polynomial.variables)
            > _MAX_TERMS
        ):
            raise ValueError("directional derivative exceeds the result-term budget")
        return self


ScalarMethod = Literal[
    "SYMPY_DIVERGENCE",
    "SYMPY_LAPLACIAN",
    "SYMPY_DIRECTIONAL_DERIVATIVE",
]
VectorMethod = Literal["SYMPY_GRADIENT", "SYMPY_CURL"]


class ScalarResult(StrictModel):
    """One canonical scalar polynomial result."""

    result: RationalPolynomial
    method: ScalarMethod


class VectorResult(StrictModel):
    """One canonical polynomial vector result."""

    components: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_POLYS
    )
    method: VectorMethod

    @model_validator(mode="after")
    def require_one_result_ring(self) -> Self:
        variables = self.components[0].variables
        if any(component.variables != variables for component in self.components):
            raise ValueError("vector result components must use one ordered ring")
        return self


__all__ = [
    "CurlRequest",
    "DirectionalDerivativeRequest",
    "ScalarFieldRequest",
    "ScalarResult",
    "VectorFieldRequest",
    "VectorResult",
]
