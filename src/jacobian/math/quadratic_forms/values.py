"""Canonical exact values for rational quadratic forms."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_QUADRATIC_FORM_DIMENSION = 32
MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS = 256
MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS = 256
MAX_QUADRATIC_EVALUATION_TERM_DIGITS = (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
    + 2 * MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS
    + 3
)
MAX_QUADRATIC_EVALUATION_DIGITS = 8_192
MAX_QUADRATIC_EVALUATION_COMMON_DENOMINATOR_DIGITS = (
    MAX_QUADRATIC_EVALUATION_DIGITS - MAX_QUADRATIC_EVALUATION_TERM_DIGITS
)


class QuadraticCrossTerm(StrictModel):
    """One nonzero coefficient of ``x_left * x_right`` with ``left < right``."""

    left: int = Field(ge=0, lt=MAX_QUADRATIC_FORM_DIMENSION)
    right: int = Field(ge=0, lt=MAX_QUADRATIC_FORM_DIMENSION)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_upper_triangular_nonzero_term(self) -> Self:
        if self.left >= self.right:
            raise ValueError("cross terms must use left < right")
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero cross terms must be omitted")
        require_bounded_rational(
            self.coefficient,
            max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
            label="quadratic-form cross coefficient",
        )
        return self


class RationalQuadraticForm(StrictModel):
    """A bounded form ``Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j`` over ``QQ``.

    ``axis`` is the ordered coordinate system.  ``diagonal_coefficients[i]``
    is ``a_i`` and each ordered cross term is ``c_ij*x_i*x_j`` for ``i < j``.
    The associated polar matrix is therefore ``B_ii=2*a_i`` and
    ``B_ij=c_ij``; its half-polar Gram matrix has diagonal ``a_i`` and
    off-diagonal ``c_ij/2``.  Neither derived matrix is independently stored.

    Axis labels must be unique, ``diagonal_coefficients`` carries exactly one
    coefficient per axis label, and every cross-term index must lie within
    the declared axis; ``require_canonical_polynomial_presentation`` enforces
    these coupled rules.
    """

    quadratic_form_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_QUADRATIC_FORM_DIMENSION,
        description=(
            "Ordered coordinate labels for the polynomial variables; "
            "labels must be unique."
        ),
    )
    diagonal_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_QUADRATIC_FORM_DIMENSION,
        description=(
            "Coefficient a_i of x_i^2 in the declared axis order; exactly "
            "one coefficient per axis label."
        ),
    )
    cross_terms: tuple[QuadraticCrossTerm, ...] = Field(
        default=(),
        max_length=MAX_QUADRATIC_FORM_DIMENSION
        * (MAX_QUADRATIC_FORM_DIMENSION - 1)
        // 2,
        description=(
            "Nonzero x_i*x_j coefficients, strictly ordered by (left, right); "
            "every cross-term index must lie within the declared axis."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_polynomial_presentation(self) -> Self:
        if len(set(self.axis)) != len(self.axis):
            raise ValueError("quadratic-form axis labels must be unique")
        if len(self.diagonal_coefficients) != len(self.axis):
            raise ValueError("diagonal coefficients must match the quadratic-form axis")
        for coefficient in self.diagonal_coefficients:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
                label="quadratic-form diagonal coefficient",
            )
        positions = tuple((term.left, term.right) for term in self.cross_terms)
        if any(right >= len(self.axis) for _, right in positions):
            raise ValueError(
                "cross-term indices must lie within the quadratic-form axis"
            )
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(
            positions
        ):
            raise ValueError("cross terms must be unique and ordered by (left, right)")
        return self


class RationalCoordinateVector(StrictModel):
    """One exact coordinate vector on an explicitly ordered rational axis.

    Axis labels must be unique, and ``coordinates`` carries exactly one
    bounded rational per label in the declared order;
    ``require_axis_bound_coordinates`` enforces these coupled rules.
    """

    vector_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_QUADRATIC_FORM_DIMENSION,
        description=(
            "Ordered labels of the rational coordinate axis; labels must "
            "be unique."
        ),
    )
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_QUADRATIC_FORM_DIMENSION,
        description=(
            "Exact values x_i in the declared axis order; exactly one "
            "coordinate per axis label."
        ),
    )

    @model_validator(mode="after")
    def require_axis_bound_coordinates(self) -> Self:
        if len(set(self.axis)) != len(self.axis):
            raise ValueError("coordinate-vector axis labels must be unique")
        if len(self.coordinates) != len(self.axis):
            raise ValueError("coordinates must match the coordinate-vector axis")
        for coordinate in self.coordinates:
            require_bounded_rational(
                coordinate,
                max_digits=MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS,
                label="quadratic-form vector coordinate",
            )
        return self


def evaluate_rational_quadratic_form(
    form: RationalQuadraticForm,
    vector: RationalCoordinateVector,
) -> Fraction:
    """Evaluate one form at an axis-matched rational vector exactly."""

    if vector.axis != form.axis:
        raise ValueError("vector axis must equal the quadratic-form axis")
    require_evaluation_budget(form, vector)
    coordinates = tuple(value.as_fraction() for value in vector.coordinates)
    value = sum(
        (
            coefficient.as_fraction() * coordinate * coordinate
            for coefficient, coordinate in zip(
                form.diagonal_coefficients, coordinates, strict=True
            )
        ),
        Fraction(),
    )
    return value + sum(
        (
            term.coefficient.as_fraction()
            * coordinates[term.left]
            * coordinates[term.right]
            for term in form.cross_terms
        ),
        Fraction(),
    )


def require_evaluation_budget(
    form: RationalQuadraticForm,
    vector: RationalCoordinateVector,
) -> None:
    """Preflight the common denominator and exact output of one evaluation.

    A common denominator divides the product of every polynomial-coefficient
    denominator and the square of every coordinate denominator.  After that
    multiplication, each numerator summand has at most the common-denominator
    digits plus one coefficient and two coordinate numerators; summing at most
    528 terms adds three decimal digits.  The bound is conservative but known
    before arithmetic and applies equally to diagonal and cross terms.
    """

    coefficient_denominator_digits = sum(
        len(coefficient.den) for coefficient in form.diagonal_coefficients
    ) + sum(len(term.coefficient.den) for term in form.cross_terms)
    coordinate_denominator_digits = sum(
        len(coordinate.den) for coordinate in vector.coordinates
    )
    common_denominator_digits = (
        coefficient_denominator_digits + 2 * coordinate_denominator_digits
    )
    if common_denominator_digits > MAX_QUADRATIC_EVALUATION_COMMON_DENOMINATOR_DIGITS:
        raise ValueError(
            "quadratic-form evaluation exceeds the aggregate denominator budget"
        )


__all__ = [
    "MAX_QUADRATIC_EVALUATION_COMMON_DENOMINATOR_DIGITS",
    "MAX_QUADRATIC_EVALUATION_DIGITS",
    "MAX_QUADRATIC_EVALUATION_TERM_DIGITS",
    "MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS",
    "MAX_QUADRATIC_FORM_DIMENSION",
    "MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS",
    "QuadraticCrossTerm",
    "RationalCoordinateVector",
    "RationalQuadraticForm",
    "evaluate_rational_quadratic_form",
    "require_evaluation_budget",
]
