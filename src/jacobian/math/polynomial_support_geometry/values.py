"""Provider-independent exact values for polynomial support geometry."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

MAX_SUPPORT_TERMS = 4096
MAX_NEWTON_DIMENSION = 8
MAX_WEIGHT_COMPONENTS = 8
# The Newton polytope operation decides each support point with an exact
# Phase-1 rational membership kernel against all other points. Worst-case
# admitted work was measured across adversarial supports (convex-position,
# dense-grid, and 8-dimensional random sets) and stays within a few seconds
# at this bound instead of inheriting the full canonical term budget.
MAX_NEWTON_TERMS = 96


class PolynomialSupport(StrictModel):
    """The exponent support of a nonzero polynomial."""

    is_zero: bool
    term_count: int = Field(ge=0)
    exponents: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_SUPPORT_TERMS
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        default=(), max_length=MAX_SUPPORT_TERMS
    )
    variables: tuple[str, ...] = Field(min_length=0, max_length=MAX_NEWTON_DIMENSION)
    coordinate_min: tuple[int, ...] = Field(default=(), max_length=MAX_NEWTON_DIMENSION)
    coordinate_max: tuple[int, ...] = Field(default=(), max_length=MAX_NEWTON_DIMENSION)
    total_degree_min: int = 0
    total_degree_max: int = 0


class NewtonPolytope(StrictModel):
    """The Newton polytope: convex hull of support exponents.

    The ordered variable axis is retained so every vertex coordinate
    denotes a specific polynomial variable and results with identical
    exponent tuples but different rings stay distinguishable.
    """

    is_zero: bool
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_NEWTON_DIMENSION)
    ambient_dimension: int = Field(ge=0)
    affine_dimension: int = Field(ge=0)
    vertices: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_SUPPORT_TERMS
    )
    nonextreme: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_SUPPORT_TERMS
    )
    all_support_exponents: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_SUPPORT_TERMS
    )


class PolynomialWeightProfile(StrictModel):
    """Weight profile of a polynomial support under a retained weight vector.

    The source polynomial and weight are retained so the minimum,
    minimizing exponents, and layers replay against their own inputs after
    serialization instead of carrying detached integer data.
    """

    polynomial: RationalPolynomial
    weight: tuple[int, ...] = Field(min_length=1)
    minimum_weight: int
    minimizing_exponents: tuple[tuple[int, ...], ...]
    weight_layers: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]

    @model_validator(mode="after")
    def bind_profile_to_source(self) -> Self:
        if len(self.weight) != len(self.polynomial.variables):
            raise ValueError("weight vector length must match variable count")
        from jacobian.math.polynomial_support_geometry.operations import (
            _compute_weight_layers,
        )

        expected = _compute_weight_layers(self.polynomial, self.weight)
        if (
            self.minimum_weight != expected[0]
            or self.minimizing_exponents != expected[1]
            or self.weight_layers != expected[2]
        ):
            raise ValueError(
                "weight profile must be the exact weighting of its "
                "retained polynomial and weight"
            )
        return self


class PolynomialFaceData(StrictModel):
    """The initial (minimum-weight) form as a canonical polynomial.

    The exposed face carries the owner-defined ``RationalPolynomial`` value
    over the source ring, alongside the retained source and weight, so it
    composes with other polynomial consumers without reattaching context.
    """

    polynomial: RationalPolynomial
    weight: tuple[int, ...] = Field(min_length=1)
    initial_form: RationalPolynomial

    @model_validator(mode="after")
    def bind_initial_form_to_source(self) -> Self:
        if len(self.weight) != len(self.polynomial.variables):
            raise ValueError("weight vector length must match variable count")
        from jacobian.math.polynomial_support_geometry.operations import (
            _initial_form_terms,
        )

        expected = _initial_form_terms(self.polynomial, self.weight)
        actual = tuple(
            (term.coefficient.as_fraction(), tuple(term.exponents))
            for term in self.initial_form.polynomial.terms
        )
        if (
            self.initial_form.variables != self.polynomial.variables
            or actual != expected
        ):
            raise ValueError(
                "initial form must be the exact minimum-weight face of its "
                "retained polynomial under the retained weight"
            )
        return self
