"""Provider-independent exact values for polynomial support geometry."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    PolynomialVariable,
    RationalPolynomial,
)

MAX_SUPPORT_TERMS = 4096
MAX_NEWTON_DIMENSION = 8
MAX_WEIGHT_COMPONENTS = 8
# The Newton polytope operation decides each support point with an exact
# Phase-1 rational membership kernel against all other points. Worst-case
# admitted work was measured across adversarial supports (convex-position,
# dense-grid, and 8-dimensional random sets) and stays within a few seconds
# at this bound instead of inheriting the full canonical term budget.
MAX_NEWTON_TERMS = 96


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial_support_geometry.{reason}", message)


def _require_canonical_exponents(exponents: tuple[tuple[int, ...], ...]) -> None:
    """Support points must lie in the canonical polynomial exponent domain."""
    if any(
        exponent < 0 or exponent > MAX_POLYNOMIAL_EXPONENT
        for exp in exponents
        for exponent in exp
    ):
        raise _validation_error(
            "exponents_out_of_domain",
            "support exponents exceed the canonical polynomial domain",
        )


def _require_nonzero_support(value: PolynomialSupport) -> None:
    """A claimed nonzero support carries nonzero coefficients and extrema."""
    # An exponent with a zero coefficient is not part of a support.
    if any(coefficient.as_fraction() == 0 for coefficient in value.coefficients):
        raise _validation_error(
            "zero_support_coefficient",
            "support coefficients must be nonzero; zero terms are omitted",
        )


def _require_shape_consistency(value: PolynomialSupport, width: int) -> None:
    """Term count, distinctness, and axis shape precede extrema replay."""
    if len(value.exponents) != len(value.coefficients) or value.term_count != len(
        value.exponents
    ):
        raise _validation_error(
            "term_count_mismatch",
            "term count must match the number of exponents and coefficients",
        )
    if len(set(value.variables)) != width:
        raise _validation_error(
            "variables_not_unique", "variables must be unique and canonically named"
        )
    # An exponent set cannot contain duplicates: a canonical polynomial
    # has unique exponent tuples, so duplicated support entries would
    # report a term count no polynomial can reconstruct.
    if len(set(value.exponents)) != len(value.exponents):
        raise _validation_error(
            "exponents_not_distinct", "support exponents must be distinct"
        )
    if any(len(exp) != width for exp in value.exponents):
        raise _validation_error(
            "exponent_dimension_mismatch",
            "exponent tuples must use the declared variable axis",
        )
    _require_canonical_exponents(value.exponents)
    if value.coordinate_min and len(value.coordinate_min) != width:
        raise _validation_error(
            "coordinate_extrema_dimension_mismatch",
            "coordinate extrema must use the declared variable axis",
        )


def _require_replayed_extrema(value: PolynomialSupport, width: int) -> None:
    """A nonzero support carries the exact extrema of its own exponents."""
    degrees = [sum(exp) for exp in value.exponents]
    if (
        value.total_degree_min != min(degrees)
        or value.total_degree_max != max(degrees)
        or tuple(value.coordinate_min)
        != tuple(min(exp[i] for exp in value.exponents) for i in range(width))
        or tuple(value.coordinate_max)
        != tuple(max(exp[i] for exp in value.exponents) for i in range(width))
    ):
        raise _validation_error(
            "extrema_mismatch",
            "coordinate and degree extrema must be the exact extrema of "
            "the retained support",
        )


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
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_NEWTON_DIMENSION
    )
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
        width = len(self.variables)
        _require_shape_consistency(self, width)
        if self.is_zero:
            if self.total_degree_min is not None or self.total_degree_max is not None:
                raise _validation_error(
                    "zero_support_degree_extrema",
                    "an empty support carries no degree extrema",
                )
            if self.coordinate_min or self.coordinate_max:
                raise _validation_error(
                    "zero_support_coordinate_extrema",
                    "an empty support carries no coordinate extrema",
                )
            if self.exponents or self.coefficients:
                raise _validation_error(
                    "zero_support_has_terms", "a zero support carries no terms"
                )
            return self
        _require_nonzero_support(self)
        if self.total_degree_min is None or self.total_degree_max is None:
            raise _validation_error(
                "nonzero_support_missing_degree_extrema",
                "a nonzero support must carry its degree extrema",
            )
        _require_replayed_extrema(self, width)
        return self


def _require_newton_context(value: NewtonPolytope) -> None:
    """Dimension context and exponent widths precede any hull replay."""
    if len(set(value.variables)) != len(value.variables):
        raise _validation_error(
            "variables_not_unique", "variables must be unique and canonically named"
        )
    if value.ambient_dimension != len(value.variables):
        raise _validation_error(
            "ambient_dimension_mismatch",
            "ambient dimension must equal the retained variable count",
        )
    if value.affine_dimension > value.ambient_dimension:
        raise _validation_error(
            "affine_dimension_exceeded",
            "affine dimension cannot exceed the ambient dimension",
        )
    # Duplicates are not a valid support or vertex set: the tuple fields are
    # authoritative, so they must be unique before the set-based partition
    # and classification checks below.
    for points in (value.vertices, value.nonextreme, value.all_support_exponents):
        if len(set(points)) != len(points):
            raise _validation_error(
                "newton_points_not_distinct",
                "retained Newton polytope points must be distinct",
            )
    if any(
        len(exp) != value.ambient_dimension
        for exp in (*value.vertices, *value.nonextreme, *value.all_support_exponents)
    ):
        raise _validation_error(
            "newton_exponent_dimension_mismatch",
            "every retained exponent must use the ambient dimension",
        )
    _require_canonical_exponents(
        (*value.vertices, *value.nonextreme, *value.all_support_exponents)
    )


class NewtonPolytope(StrictModel):
    """The Newton polytope: convex hull of support exponents.

    The ordered variable axis is retained so every vertex coordinate
    denotes a specific polynomial variable and results with identical
    exponent tuples but different rings stay distinguishable.
    """

    is_zero: bool
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_NEWTON_DIMENSION
    )
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
        # The producer's empty-polytope convention is affine dimension
        # zero; a deserialized zero result must not contradict its own
        # empty support.
        if self.is_zero and self.affine_dimension != 0:
            raise _validation_error(
                "zero_newton_affine_dimension",
                (
                    "the zero polynomial's empty Newton polytope has affine dimension zero"
                ),
            )
        if not self.is_zero:
            support = set(self.all_support_exponents)
            if support != set(self.vertices) | set(self.nonextreme):
                raise _validation_error(
                    "newton_support_partition_mismatch",
                    (
                        "vertices and nonextreme points must partition the retained support"
                    ),
                )
            if set(self.vertices) & set(self.nonextreme):
                raise _validation_error(
                    "newton_vertex_overlap",
                    "an exponent cannot be both a vertex and non-extreme",
                )
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
                raise _validation_error(
                    "newton_vertex_classification_mismatch",
                    (
                        "vertex classification must be the exact convex-hull "
                        "classification of the retained support"
                    ),
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
                raise _validation_error(
                    "newton_affine_dimension_mismatch",
                    (
                        f"affine_dimension {self.affine_dimension} must equal "
                        f"the hull's exact affine dimension {dimension}"
                    ),
                )
        elif self.vertices or self.nonextreme or self.all_support_exponents:
            raise _validation_error(
                "zero_newton_has_points",
                "the zero polynomial has an empty Newton polytope",
            )
        if not self.is_zero and not self.all_support_exponents:
            raise _validation_error(
                "nonzero_newton_missing_support",
                (
                    "a nonzero Newton polytope must retain its support so the "
                    "classification stays reconstructible"
                ),
            )
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
            raise _validation_error(
                "weight_dimension_mismatch",
                "weight vector length must match variable count",
            )
        from jacobian.math.polynomial_support_geometry.operations import (
            _compute_weight_layers,
        )

        expected = _compute_weight_layers(self.polynomial, self.weight)
        if (
            self.minimum_weight != expected[0]
            or self.minimizing_exponents != expected[1]
            or self.weight_layers != expected[2]
        ):
            raise _validation_error(
                "weight_profile_mismatch",
                (
                    "weight profile must be the exact weighting of its "
                    "retained polynomial and weight"
                ),
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
            raise _validation_error(
                "weight_dimension_mismatch",
                "weight vector length must match variable count",
            )
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
            raise _validation_error(
                "initial_form_mismatch",
                (
                    "initial form must be the exact minimum-weight face of its "
                    "retained polynomial under the retained weight"
                ),
            )
        return self
