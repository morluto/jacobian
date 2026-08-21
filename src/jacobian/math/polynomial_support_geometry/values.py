"""Provider-independent exact values for polynomial support geometry."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_SUPPORT_TERMS = 4096
MAX_NEWTON_DIMENSION = 8
MAX_WEIGHT_COMPONENTS = 8


class PolynomialSupport(StrictModel):
    """The exponent support of a nonzero polynomial."""

    is_zero: bool
    term_count: int = Field(ge=0)
    exponents: tuple[tuple[int, ...], ...] = Field(default=(), max_length=MAX_SUPPORT_TERMS)
    coefficients: tuple[CanonicalRational, ...] = Field(default=(), max_length=MAX_SUPPORT_TERMS)
    variables: tuple[str, ...] = Field(min_length=0, max_length=MAX_NEWTON_DIMENSION)
    coordinate_min: tuple[int, ...] = Field(default=(), max_length=MAX_NEWTON_DIMENSION)
    coordinate_max: tuple[int, ...] = Field(default=(), max_length=MAX_NEWTON_DIMENSION)
    total_degree_min: int = 0
    total_degree_max: int = 0


class NewtonPolytope(StrictModel):
    """The Newton polytope: convex hull of support exponents."""

    is_zero: bool
    ambient_dimension: int = Field(ge=0)
    affine_dimension: int = Field(ge=0)
    vertices: tuple[tuple[int, ...], ...] = Field(default=(), max_length=MAX_SUPPORT_TERMS)
    nonextreme: tuple[tuple[int, ...], ...] = Field(default=(), max_length=MAX_SUPPORT_TERMS)
    all_support_exponents: tuple[tuple[int, ...], ...] = Field(default=(), max_length=MAX_SUPPORT_TERMS)


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
