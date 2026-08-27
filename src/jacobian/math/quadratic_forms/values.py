"""Canonical exact values for rational quadratic forms."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS = 256
MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS = 256
MAX_QUADRATIC_EVALUATION_TERM_DIGITS = (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS + 2 * MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS
)
MAX_QUADRATIC_EVALUATION_DIGITS = 8_192
MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.{reason}", message)


def _require_bounded(
    value: CanonicalRational, *, max_digits: int, label: str, reason: str
) -> None:
    try:
        require_bounded_rational(value, max_digits=max_digits, label=label)
    except ValueError as error:
        raise _validation_error(reason, str(error)) from error


class QuadraticCrossTerm(StrictModel):
    """One nonzero coefficient of ``x_left * x_right`` with ``left < right``."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)
    coefficient: CanonicalRational = Field(
        description=(
            f"Nonzero rational multiplier of x_left * x_right; numerator "
            f"and denominator carry at most "
            f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS} decimal digits."
        )
    )

    @model_validator(mode="after")
    def require_upper_triangular_nonzero_term(self) -> Self:
        if self.left >= self.right:
            raise _validation_error(
                "cross_term_order", "cross terms must use left < right"
            )
        if self.coefficient.as_fraction() == 0:
            raise _validation_error(
                "zero_cross_term", "zero cross terms must be omitted"
            )
        _require_bounded(
            self.coefficient,
            max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
            label="quadratic-form cross coefficient",
            reason="coefficient_bound",
        )
        return self


class RationalQuadraticForm(StrictModel):
    """A bounded form ``Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j`` over ``QQ``.

    ``axis`` is the ordered coordinate system.  ``diagonal_coefficients[i]``
    is ``a_i`` and each ordered cross term is ``c_ij*x_i*x_j`` for ``i < j``.
    The associated polar matrix is therefore ``B_ii=2*a_i`` and
    ``B_ij=c_ij``; its half-polar Gram matrix has diagonal ``a_i`` and
    off-diagonal ``c_ij/2``.  Neither derived matrix is independently stored.
    The empty axis is the unique zero-dimensional form, whose evaluation is
    always ``0``.

    Axis labels must be unique, ``diagonal_coefficients`` carries exactly one
    coefficient per axis label, and every cross-term index must lie within
    the declared axis; ``require_canonical_polynomial_presentation`` enforces
    these coupled rules.
    """

    domain: Literal["QQ"] = "QQ"
    axis: tuple[OpaqueLabel, ...] = Field(
        description=(
            "Ordered coordinate labels for the polynomial variables; "
            "labels must be unique."
        ),
    )
    diagonal_coefficients: tuple[CanonicalRational, ...] = Field(
        description=(
            "Coefficient a_i of x_i^2 in the declared axis order; exactly "
            f"one coefficient per axis label, and each numerator and "
            f"denominator carries at most "
            f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS} decimal digits."
        ),
    )
    cross_terms: tuple[QuadraticCrossTerm, ...] = Field(
        default=(),
        description=(
            "Nonzero x_i*x_j coefficients, strictly ordered by (left, right); "
            "every cross-term index must lie within the declared axis and "
            f"each coefficient numerator and denominator carries at most "
            f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS} decimal digits."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_polynomial_presentation(self) -> Self:
        if len(set(self.axis)) != len(self.axis):
            raise _validation_error(
                "axis_labels_not_unique", "quadratic-form axis labels must be unique"
            )
        if len(self.diagonal_coefficients) != len(self.axis):
            raise _validation_error(
                "diagonal_length_mismatch",
                "diagonal coefficients must match the quadratic-form axis",
            )
        for coefficient in self.diagonal_coefficients:
            _require_bounded(
                coefficient,
                max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
                label="quadratic-form diagonal coefficient",
                reason="coefficient_bound",
            )
        positions = tuple((term.left, term.right) for term in self.cross_terms)
        if any(right >= len(self.axis) for _, right in positions):
            raise _validation_error(
                "cross_term_out_of_range",
                "cross-term indices must lie within the quadratic-form axis",
            )
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(
            positions
        ):
            raise _validation_error(
                "cross_terms_not_canonical",
                "cross terms must be unique and ordered by (left, right)",
            )
        return self


class RationalCoordinateVector(StrictModel):
    """One exact coordinate vector on an explicitly ordered rational axis.

    Axis labels must be unique, and ``coordinates`` carries exactly one
    bounded rational per label in the declared order; the empty axis is the
    unique zero-dimensional vector.
    ``require_axis_bound_coordinates`` enforces these coupled rules.
    """

    domain: Literal["QQ"] = "QQ"
    axis: tuple[OpaqueLabel, ...] = Field(
        description=(
            "Ordered labels of the rational coordinate axis; labels must be unique."
        ),
    )
    coordinates: tuple[CanonicalRational, ...] = Field(
        description=(
            "Exact values x_i in the declared axis order; exactly one "
            "coordinate per axis label, and each numerator and denominator "
            f"carries at most {MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS} "
            f"decimal digits."
        ),
    )

    @model_validator(mode="after")
    def require_axis_bound_coordinates(self) -> Self:
        if len(set(self.axis)) != len(self.axis):
            raise _validation_error(
                "axis_labels_not_unique", "coordinate-vector axis labels must be unique"
            )
        if len(self.coordinates) != len(self.axis):
            raise _validation_error(
                "coordinate_length_mismatch",
                "coordinates must match the coordinate-vector axis",
            )
        for coordinate in self.coordinates:
            _require_bounded(
                coordinate,
                max_digits=MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS,
                label="quadratic-form vector coordinate",
                reason="coordinate_bound",
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
    """Preflight total support, common denominator, and exact output size.

    Only monomials with a nonzero coefficient evaluated at nonzero
    coordinates contribute to ``Q(vector)``, so every arithmetic budget is
    based on those active monomials alone.  A common denominator divides the
    product of the active polynomial-coefficient denominators and the squares
    of the denominators of the coordinates those monomials touch, so its digit
    length is at most the aggregate ``d`` of those digit lengths.  Over that
    shared denominator each numerator summand carries one coefficient and
    two coordinate numerators on top of ``d``, hence at most
    ``d + MAX_QUADRATIC_EVALUATION_TERM_DIGITS`` digits, and summing the
    ``t`` active monomials adds at most the decimal digits of ``t``.  The
    reduced value therefore keeps both components at or below
    ``MAX_QUADRATIC_EVALUATION_DIGITS`` digits whenever

        d + MAX_QUADRATIC_EVALUATION_TERM_DIGITS + len(str(t))
            <= MAX_QUADRATIC_EVALUATION_DIGITS,

    which this preflight rejects against before any arithmetic runs.  The
    bound is conservative but known before execution and admits light forms
    at every dimension.

    The active accounting deliberately ignores annihilated monomials, so it
    says nothing about how much support a request may materialize.
    Request admission, kernel traversal, and source-bound serialization all
    visit every stored diagonal coefficient, cross term,
    and coordinate whether or not it contributes, and each stored entry has
    bounded height (per-entry numerator/denominator digit bounds; bounded
    label length).  The total materialized form support,

        len(diagonal_coefficients) + len(cross_terms),

    is therefore capped at ``MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS`` before
    the active accounting runs.  The coupled length validators keep the
    coordinate count equal to the diagonal-coefficient count, so this single
    bound also limits every linear traversal and the request echoed inside
    the serialized result, independently of the arithmetic-digit budgets
    above.
    """

    support_terms = len(form.diagonal_coefficients) + len(form.cross_terms)
    if support_terms > MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS:
        raise ValueError("quadratic-form evaluation exceeds the total support budget")
    nonzero_coordinates = {
        index
        for index, coordinate in enumerate(vector.coordinates)
        if coordinate.as_fraction() != 0
    }
    active_coordinates: set[int] = set()
    terms = 0
    common_denominator_digits = 0
    for index, coefficient in enumerate(form.diagonal_coefficients):
        if index in nonzero_coordinates and coefficient.as_fraction() != 0:
            terms += 1
            common_denominator_digits += len(coefficient.den)
            active_coordinates.add(index)
    for term in form.cross_terms:
        if term.left in nonzero_coordinates and term.right in nonzero_coordinates:
            terms += 1
            common_denominator_digits += len(term.coefficient.den)
            active_coordinates.update((term.left, term.right))
    common_denominator_digits += 2 * sum(
        len(vector.coordinates[index].den) for index in sorted(active_coordinates)
    )
    if (
        common_denominator_digits
        + MAX_QUADRATIC_EVALUATION_TERM_DIGITS
        + len(str(terms))
        > MAX_QUADRATIC_EVALUATION_DIGITS
    ):
        raise ValueError(
            "quadratic-form evaluation exceeds the aggregate denominator budget"
        )


__all__ = [
    "MAX_QUADRATIC_EVALUATION_DIGITS",
    "MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS",
    "MAX_QUADRATIC_EVALUATION_TERM_DIGITS",
    "MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS",
    "MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS",
    "QuadraticCrossTerm",
    "RationalCoordinateVector",
    "RationalQuadraticForm",
    "evaluate_rational_quadratic_form",
    "require_evaluation_budget",
]
