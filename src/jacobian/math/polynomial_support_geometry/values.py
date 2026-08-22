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
    # Empty support has no degree extrema; they are None exactly when the
    # polynomial is zero instead of fabricating the constant polynomial's 0.
    total_degree_min: int | None = None
    total_degree_max: int | None = None

    @model_validator(mode="after")
    def bind_extrema_to_support(self) -> Self:
        # Cross-field consistency: the term count, exponent tuples, and
        # coefficients must agree, and the coordinate/degree extrema must
        # match the retained support.
        if len(self.exponents) != len(self.coefficients) or self.term_count != len(
            self.exponents
        ):
            raise ValueError(
                "term count must match the number of exponents and coefficients"
            )
        width = len(self.variables)
        if any(len(exp) != width for exp in self.exponents):
            raise ValueError("exponent tuples must use the declared variable axis")
        if self.coordinate_min and len(self.coordinate_min) != width:
            raise ValueError("coordinate extrema must use the declared variable axis")
        if self.is_zero:
            if self.total_degree_min is not None or self.total_degree_max is not None:
                raise ValueError("an empty support carries no degree extrema")
            if self.exponents or self.coefficients:
                raise ValueError("a zero support carries no terms")
            return self
        if self.total_degree_min is None or self.total_degree_max is None:
            raise ValueError("a nonzero support must carry its degree extrema")
        degrees = [sum(exp) for exp in self.exponents]
        if (
            self.total_degree_min != min(degrees)
            or self.total_degree_max != max(degrees)
            or tuple(self.coordinate_min)
            != tuple(min(exp[i] for exp in self.exponents) for i in range(width))
            or tuple(self.coordinate_max)
            != tuple(max(exp[i] for exp in self.exponents) for i in range(width))
        ):
            raise ValueError(
                "coordinate and degree extrema must be the exact extrema of "
                "the retained support"
            )
        return self


def _require_newton_context(value: NewtonPolytope) -> None:
    """Dimension context and exponent widths precede any hull replay."""
    if value.ambient_dimension != len(value.variables):
        raise ValueError("ambient dimension must equal the retained variable count")
    if value.affine_dimension > value.ambient_dimension:
        raise ValueError("affine dimension cannot exceed the ambient dimension")
    if any(
        len(exp) != value.ambient_dimension
        for exp in (*value.vertices, *value.nonextreme, *value.all_support_exponents)
    ):
        raise ValueError("every retained exponent must use the ambient dimension")


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
    # Retained support fields carry the admitted hull size (96 points): the
    # exact replay below runs one convex-membership LP per point, so a
    # deserialized payload must not bypass the operation's work admission.
    vertices: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_NEWTON_TERMS
    )
    nonextreme: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_NEWTON_TERMS
    )
    all_support_exponents: tuple[tuple[int, ...], ...] = Field(
        default=(), max_length=MAX_NEWTON_TERMS
    )

    @model_validator(mode="after")
    def require_newton_invariants(self) -> Self:
        _require_newton_context(self)
        if not self.is_zero:
            support = set(self.all_support_exponents)
            if support != set(self.vertices) | set(self.nonextreme):
                raise ValueError(
                    "vertices and nonextreme points must partition the retained support"
                )
            if set(self.vertices) & set(self.nonextreme):
                raise ValueError("an exponent cannot be both a vertex and non-extreme")
            # Exact replay: the classification must be the true hull of the
            # retained support (bounded by the 96-term admission).
            from jacobian.math.polynomial_support_geometry.operations import (
                _is_vertex,
                _matrix_rank,
            )

            replayed_vertices = [
                exp
                for exp in self.all_support_exponents
                if _is_vertex(exp, [q for q in self.all_support_exponents if q != exp])
            ]
            replayed_nonextreme = [
                exp
                for exp in self.all_support_exponents
                if exp not in set(replayed_vertices)
            ]
            if set(self.vertices) != set(replayed_vertices) or set(
                self.nonextreme
            ) != set(replayed_nonextreme):
                raise ValueError(
                    "vertex classification must be the exact convex-hull "
                    "classification of the retained support"
                )
            if len(replayed_vertices) > 1:
                first = replayed_vertices[0]
                dimension = _matrix_rank(
                    [
                        [v[j] - first[j] for j in range(len(first))]
                        for v in replayed_vertices[1:]
                    ]
                )
            else:
                dimension = 0
            if self.affine_dimension != dimension:
                raise ValueError(
                    f"affine_dimension {self.affine_dimension} must equal "
                    f"the hull's exact affine dimension {dimension}"
                )
        elif self.vertices or self.nonextreme or self.all_support_exponents:
            raise ValueError("the zero polynomial has an empty Newton polytope")
        return self


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
