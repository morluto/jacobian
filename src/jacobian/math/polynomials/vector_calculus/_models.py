"""Typed wire contracts for polynomial vector calculus operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
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


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by polynomial-vector contracts."""

    return PydanticCustomError(f"polynomial_vector_calc.{reason}", message)


def _require_field_polynomial(
    polynomial: RationalPolynomial,
    *,
    label: str,
) -> None:
    if len(polynomial.variables) > MAX_VARS:
        raise _validation_error(
            "variable_budget", f"{label} exceeds the {MAX_VARS}-variable budget"
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_TERMS,
        maximum_exponent=_MAX_EXPONENT,
        maximum_coefficient_digits=_MAX_COEFFICIENT_DIGITS,
        label=label,
    )
    if any(sum(term.exponents) > _MAX_EXPONENT for term in polynomial.polynomial.terms):
        raise _validation_error(
            "total_degree", f"{label} exceeds total degree {_MAX_EXPONENT}"
        )


class ScalarFieldRequest(StrictModel):
    """One bounded canonical multivariate polynomial scalar field."""

    polynomial: RationalPolynomial


class VectorFieldRequest(StrictModel):
    """A polynomial vector field with one component per ordered variable."""

    components: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_POLYS
    )

    @model_validator(mode="after")
    def require_one_vector_field_ring(self) -> Self:
        variables = self.components[0].variables
        if len(self.components) != len(variables):
            raise _validation_error(
                "component_count", "vector field must have one component per variable"
            )
        for component in self.components:
            if component.variables != variables:
                raise _validation_error(
                    "ordered_ring", "vector-field components must use one ordered ring"
                )
        return self


class CurlRequest(VectorFieldRequest):
    """A three-dimensional polynomial vector field."""

    @model_validator(mode="after")
    def require_three_dimensions(self) -> Self:
        if len(self.components[0].variables) != 3:
            raise _validation_error(
                "curl_dimensions",
                "curl requires exactly three variables and components",
            )
        return self


class DirectionalDerivativeRequest(StrictModel):
    """Directional derivative along one exact constant vector."""

    polynomial: RationalPolynomial
    direction: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=MAX_VARS)

    @model_validator(mode="after")
    def require_matching_bounded_direction(self) -> Self:
        if len(self.direction) != len(self.polynomial.variables):
            raise _validation_error(
                "direction_length",
                "direction vector length must match the polynomial axis",
            )
        return self


class ScalarResult(StrictModel):
    """One canonical scalar polynomial result."""

    result: RationalPolynomial


class VectorResult(StrictModel):
    """One canonical polynomial vector result."""

    components: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_POLYS
    )

    @model_validator(mode="after")
    def require_one_result_ring(self) -> Self:
        variables = self.components[0].variables
        if any(component.variables != variables for component in self.components):
            raise _validation_error(
                "ordered_ring", "vector result components must use one ordered ring"
            )
        return self


__all__ = [
    "CurlRequest",
    "DirectionalDerivativeRequest",
    "ScalarFieldRequest",
    "ScalarResult",
    "VectorFieldRequest",
    "VectorResult",
]
