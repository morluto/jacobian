"""Exact rational quadratic-form operations."""

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalCoordinateVector,
    RationalQuadraticForm,
    require_evaluation_budget,
)


def evaluate_rational_quadratic_form(
    form: RationalQuadraticForm,
    vector: RationalCoordinateVector,
) -> Fraction:
    """Evaluate one admitted form at an axis-matched rational vector exactly."""

    if vector.axis != form.axis:
        raise OperationDomainValidationError(
            location=("vector", "axis"),
            code="quadratic_form.axis_mismatch",
            message="vector axis must equal the quadratic-form axis",
        )
    try:
        require_evaluation_budget(form, vector)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("form", "vector"), code=exc.type, message=exc.message()
        ) from exc
    coordinates = tuple(value.as_fraction() for value in vector.coordinates)
    diagonal = sum(
        (
            coefficient.as_fraction() * coordinate * coordinate
            for coefficient, coordinate in zip(
                form.diagonal_coefficients, coordinates, strict=True
            )
        ),
        Fraction(),
    )
    cross = sum(
        (
            term.coefficient.as_fraction()
            * coordinates[term.left]
            * coordinates[term.right]
            for term in form.cross_terms
        ),
        Fraction(),
    )
    return diagonal + cross


__all__ = ["evaluate_rational_quadratic_form"]
