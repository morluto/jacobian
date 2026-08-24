"""Canonical bounded rational polynomial-map values."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_MAP_INPUTS = 8
MAX_MAP_OUTPUTS = 20
MAX_MAP_POLYNOMIAL_TERMS = 256
MAX_MAP_POLYNOMIAL_EXPONENT = 64
MAX_MAP_COEFFICIENT_DIGITS = 128


def require_map_polynomial(polynomial: RationalPolynomial, *, label: str) -> None:
    """Apply the shared polynomial-map representation budget."""

    if len(polynomial.variables) > MAX_MAP_INPUTS:
        raise ValueError(f"{label} exceeds the {MAX_MAP_INPUTS}-variable budget")
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_MAP_POLYNOMIAL_TERMS,
        maximum_exponent=MAX_MAP_POLYNOMIAL_EXPONENT,
        maximum_coefficient_digits=MAX_MAP_COEFFICIENT_DIGITS,
        label=label,
    )
    if any(
        sum(term.exponents) > MAX_MAP_POLYNOMIAL_EXPONENT
        for term in polynomial.polynomial.terms
    ):
        raise ValueError(f"{label} exceeds total degree {MAX_MAP_POLYNOMIAL_EXPONENT}")


class RationalPolynomialMap(StrictModel):
    """An ordered tuple of polynomials defining ``A^n_QQ -> A^m_QQ``.

    Coordinate positions are the target axis. Every component belongs to the
    same explicitly ordered source polynomial ring.
    """

    polynomial_map_schema_version: Literal["1"] = "1"
    input_variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_MAP_INPUTS,
        description="Ordered source-coordinate axis of the polynomial map.",
    )
    output_polynomials: tuple[RationalPolynomial, ...] = Field(
        min_length=1,
        max_length=MAX_MAP_OUTPUTS,
        description=(
            "Ordered target-coordinate tuple; every polynomial must use the "
            "complete ordered input_variables ring."
        ),
    )

    @model_validator(mode="after")
    def require_one_map_ring(self) -> Self:
        if len(set(self.input_variables)) != len(self.input_variables):
            raise ValueError("input variables must be unique")
        for polynomial in self.output_polynomials:
            require_map_polynomial(polynomial, label="map output polynomial")
            if polynomial.variables != self.input_variables:
                raise ValueError(
                    "every map output must use the complete ordered input axis"
                )
        return self


__all__ = ["RationalPolynomialMap"]
