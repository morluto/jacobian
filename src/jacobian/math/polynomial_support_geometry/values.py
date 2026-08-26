"""Provider-independent exact values for polynomial support geometry."""

from __future__ import annotations

from typing import Any, Self

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
MAX_WEIGHT_PROFILE_TERMS = 1024
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
    """A claimed nonzero support carries only nonzero coefficients."""
    # An exponent with a zero coefficient is not part of a support.
    if any(coefficient.as_fraction() == 0 for coefficient in value.coefficients):
        raise _validation_error(
            "zero_support_coefficient",
            "support coefficients must be nonzero; zero terms are omitted",
        )


def _require_shape_consistency(value: PolynomialSupport, width: int) -> None:
    """Term count, distinctness, and axis shape are structural invariants."""
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
        # Parsing checks the self-contained canonical shape only.  Computing
        # extrema is deliberately left to ``verify_polynomial_support`` so
        # deserializing a result never re-enters an operation kernel.
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
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build a support value from the trusted owner-local kernel."""
        return cls(**values)


def _require_newton_context(value: NewtonPolytope) -> None:
    """Dimension context and exponent widths bound an independent verifier."""
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
    # and partition checks below.
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
    # Retained support fields carry the admitted hull size (96 points), so an
    # explicit verifier cannot be asked to replay an unbounded hull claim.
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

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build a Newton polytope from trusted owner-local kernel output."""
        return cls(**values)


class PolynomialWeightProfile(StrictModel):
    """Weight profile of a polynomial support under a retained weight vector.

    The source polynomial and weight keep the claimed profile composable and
    make its defining identity available to an explicit bounded verifier.
    """

    polynomial: RationalPolynomial
    weight: tuple[int, ...] = Field(min_length=1)
    minimum_weight: int
    minimizing_exponents: tuple[tuple[int, ...], ...] = Field(
        max_length=MAX_WEIGHT_PROFILE_TERMS
    )
    weight_layers: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...] = Field(
        max_length=MAX_WEIGHT_PROFILE_TERMS
    )

    @model_validator(mode="after")
    def bind_profile_to_source(self) -> Self:
        if len(self.weight) != len(self.polynomial.variables):
            raise _validation_error(
                "weight_dimension_mismatch",
                "weight vector length must match variable count",
            )
        if any(
            len(exponents) > MAX_WEIGHT_COMPONENTS
            for exponents in self.minimizing_exponents
        ) or any(
            len(exponents) > MAX_WEIGHT_COMPONENTS
            for _, layer in self.weight_layers
            for exponents in layer
        ):
            raise _validation_error(
                "weight_profile_exponent_dimension_exceeded",
                "weight-profile exponents exceed the canonical variable-axis bound",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build a weight profile from trusted owner-local kernel output."""
        return cls(**values)


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
        if self.initial_form.variables != self.polynomial.variables:
            raise _validation_error(
                "initial_form_variable_mismatch",
                ("initial form must use the source polynomial's variable axis"),
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build an initial-form value from trusted owner-local kernel output."""
        return cls(**values)
