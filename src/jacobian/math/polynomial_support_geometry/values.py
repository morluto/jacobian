"""Provider-independent exact values for polynomial support geometry."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

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
    """Weight profile of a polynomial support under a weight vector."""

    minimum_weight: int
    minimizing_exponents: tuple[tuple[int, ...], ...]
    weight_layers: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]


class PolynomialFaceData(StrictModel):
    """exponents on the exposed face of the Newton polytope."""

    face_exponents: tuple[tuple[int, ...], ...]
    face_coefficients: tuple[CanonicalRational, ...]
    face_polynomial_terms: tuple[dict, ...]
