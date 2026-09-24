"""Typed contracts for bounded diagonal multiplicative-group actions."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalLaurentPolynomial,
    RationalPolynomial,
    SparseRationalPolynomial,
)

MAX_DIAGONAL_WEIGHT = 64
MAX_WEIGHT_ACTION_TERMS = 256
# The Laurent coaction carrier reserves its eighth axis for the parameter, so
# the source ring of a weight action is bounded like the Ga-action carrier.
MAX_WEIGHT_ACTION_VARIABLES = 7
MAX_WEIGHT_ACTION_DEGREE = 64
MAX_GM_INVARIANT_DEGREE = 64
MAX_GM_INVARIANT_MONOMIALS = 4_096


class PolynomialWeightAction(StrictModel):
    """The diagonal action ``lambda.x_i = lambda**weights[i] * x_i`` on QQ[R]."""

    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=8)
    weights: tuple[StrictInt, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_complete_bounded_weights(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("weight-action variables must be unique")
        if len(self.variables) != len(self.weights):
            raise ValueError("one integer weight is required per variable")
        if any(abs(weight) > MAX_DIAGONAL_WEIGHT for weight in self.weights):
            raise ValueError(f"variable weights are bounded by {MAX_DIAGONAL_WEIGHT}")
        return self


class PolynomialWeightActionRequest(StrictModel):
    action: PolynomialWeightAction
    polynomial: RationalPolynomial
    parameter: PolynomialVariable = "t"

    @model_validator(mode="after")
    def require_bound_parent(self) -> Self:
        if self.polynomial.variables != self.action.variables:
            raise ValueError("polynomial and action must use the same ordered QQ ring")
        if self.parameter in self.action.variables:
            raise ValueError(
                "the Laurent parameter must be distinct from ring variables"
            )
        if len(self.action.variables) > MAX_WEIGHT_ACTION_VARIABLES:
            raise ValueError(
                "the diagonal action is bounded to "
                f"{MAX_WEIGHT_ACTION_VARIABLES} source variables because the "
                "Laurent coaction carrier reserves its eighth axis for the "
                "parameter"
            )
        return self


class PolynomialWeightComponent(StrictModel):
    weight: StrictInt
    polynomial: RationalPolynomial


class PolynomialWeightActionResult(StrictModel):
    """Exact coaction image, weight split, and fixed slice of one source."""

    action: PolynomialWeightAction
    source: RationalPolynomial
    parameter: PolynomialVariable
    coaction: RationalLaurentPolynomial
    components: tuple[PolynomialWeightComponent, ...]
    weight_zero: RationalPolynomial

    @model_validator(mode="after")
    def require_parent_and_projection(self) -> Self:
        if self.source.variables != self.action.variables:
            raise ValueError("result source must remain in the action's ordered ring")
        if self.coaction.variables != (*self.action.variables, self.parameter):
            raise ValueError(
                "coaction must use the action ring extended by the parameter"
            )
        if self.weight_zero.variables != self.action.variables:
            raise ValueError("weight-zero projection must retain the source ring")
        if tuple(c.weight for c in self.components) != tuple(
            sorted(c.weight for c in self.components)
        ):
            raise ValueError("weight components must be ordered by increasing weight")
        if any(
            c.polynomial.variables != self.action.variables for c in self.components
        ):
            raise ValueError("weight components must retain the action's ordered ring")
        zero = next((c.polynomial for c in self.components if c.weight == 0), None)
        if zero is None:
            zero = RationalPolynomial(
                variables=self.action.variables,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        if zero != self.weight_zero:
            raise ValueError(
                "weight-zero projection must equal the zero-weight component"
            )
        return self


class PolynomialWeightInvariantRequest(StrictModel):
    """Request the exact invariant slice of the total-degree truncation."""

    action: PolynomialWeightAction
    degree: StrictInt = Field(ge=0, le=MAX_GM_INVARIANT_DEGREE)


class PolynomialWeightDegreeDimension(StrictModel):
    degree: StrictInt = Field(ge=0, le=MAX_GM_INVARIANT_DEGREE)
    dimension: StrictInt = Field(ge=0, le=MAX_GM_INVARIANT_MONOMIALS)


class PolynomialWeightInvariantResult(StrictModel):
    """Canonical monomial basis and Hilbert prefix for the weight-zero slice."""

    action: PolynomialWeightAction
    degree: StrictInt = Field(ge=0, le=MAX_GM_INVARIANT_DEGREE)
    basis: tuple[RationalPolynomial, ...] = Field(max_length=MAX_GM_INVARIANT_MONOMIALS)
    hilbert_prefix: tuple[PolynomialWeightDegreeDimension, ...] = Field(
        min_length=1, max_length=MAX_GM_INVARIANT_DEGREE + 1
    )
    dimension: StrictInt = Field(ge=0, le=MAX_GM_INVARIANT_MONOMIALS)

    @model_validator(mode="after")
    def require_exact_parent_and_hilbert_profile(self) -> Self:
        if len(self.hilbert_prefix) != self.degree + 1:
            raise ValueError(
                "Hilbert prefix must contain every degree from zero through the bound"
            )
        if tuple(row.degree for row in self.hilbert_prefix) != tuple(
            range(self.degree + 1)
        ):
            raise ValueError("Hilbert prefix degrees must be complete and ordered")
        if self.dimension != sum(row.dimension for row in self.hilbert_prefix):
            raise ValueError("invariant dimension must sum the Hilbert prefix")
        if any(
            polynomial.variables != self.action.variables for polynomial in self.basis
        ):
            raise ValueError("invariant basis polynomials must retain the action ring")
        if len(self.basis) != self.dimension:
            raise ValueError(
                "invariant monomial basis size must equal the exact dimension"
            )
        return self
