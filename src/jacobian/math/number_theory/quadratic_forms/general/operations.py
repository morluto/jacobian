"""Exact rational quadratic-form operations."""

from fractions import Fraction
from math import gcd

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions
from jacobian.math.number_theory.quadratic_forms.general._models import (
    MAX_COEFFICIENT_MATRIX_AXIS,
    IntegralContentRequest,
    IntegralContentResult,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
    require_bilinear_pairing_budget,
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


def bilinear_pairing(
    form: RationalQuadraticForm,
    left: RationalCoordinateVector,
    right: RationalCoordinateVector,
) -> Fraction:
    """Compute ``Q(left+right)-Q(left)-Q(right)`` exactly on the shared axis."""

    if left.axis != form.axis or right.axis != form.axis:
        raise OperationDomainValidationError(
            location=("left", "right", "axis"),
            code="quadratic_form.axis_mismatch",
            message="both vectors must use the quadratic-form axis",
        )
    try:
        require_bilinear_pairing_budget(form, left, right)
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=("form", "left", "right"),
            code="quadratic_form.pairing_budget",
            message=str(exc),
        ) from exc
    x = tuple(value.as_fraction() for value in left.coordinates)
    y = tuple(value.as_fraction() for value in right.coordinates)
    diagonal = sum(
        (
            2 * coefficient.as_fraction() * x[index] * y[index]
            for index, coefficient in enumerate(form.diagonal_coefficients)
        ),
        Fraction(),
    )
    cross = sum(
        (
            term.coefficient.as_fraction()
            * (x[term.left] * y[term.right] + x[term.right] * y[term.left])
            for term in form.cross_terms
        ),
        Fraction(),
    )
    return diagonal + cross


def require_coefficient_matrix_budget(form: RationalQuadraticForm) -> None:
    """Preflight the dense symmetric-matrix output envelope once per call.

    The kernel writes exactly ``n * n`` rationals for a form of dimension
    ``n``; every entry is a stored diagonal coefficient or half of a stored
    cross-term coefficient, so per-entry digits stay within one digit of the
    form's own coefficient bound. Bounding ``n`` before allocation bounds
    both the quadratic kernel traversal and the serialized result.
    """

    if not isinstance(form, RationalQuadraticForm):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.coefficient_matrix_form_type",
            message="form must be a rational quadratic form value",
        )
    dimension = len(form.axis)
    if dimension > MAX_COEFFICIENT_MATRIX_AXIS:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.coefficient_matrix_axis_bound",
            message=(
                "quadratic-form dimension exceeds the "
                f"{MAX_COEFFICIENT_MATRIX_AXIS}-axis coefficient-matrix envelope"
            ),
        )


def coefficient_matrix_entries(
    form: RationalQuadraticForm,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the exact symmetric matrix with ``Q(x) = x^T A x``.

    Diagonal entry ``(i, i)`` is the ``x_i^2`` polynomial coefficient and
    off-diagonal entries ``(i, j)``/``(j, i)`` are half the ``x_i*x_j``
    polynomial coefficient, so odd cross terms yield half-integral entries.
    """

    require_coefficient_matrix_budget(form)
    dimension = len(form.axis)
    entries: list[list[Fraction]] = [
        [Fraction(0) for _ in range(dimension)] for _ in range(dimension)
    ]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        entries[index][index] = coefficient.as_fraction()
    for term in form.cross_terms:
        half = term.coefficient.as_fraction() / 2
        entries[term.left][term.right] = half
        entries[term.right][term.left] = half
    return tuple(tuple(row) for row in entries)


def coefficient_matrix(form: RationalQuadraticForm) -> RationalMatrix:
    """Build the exact symmetric ``RationalMatrix`` with ``Q(x) = x^T A x``."""

    entries = coefficient_matrix_entries(form)
    return rational_matrix_from_fractions(entries)


def integral_coefficient_content(
    request: IntegralContentRequest,
) -> IntegralContentResult:
    """Return gcd of integral polynomial coefficients and their primitive quotient.

    The zero polynomial has content zero and is returned unchanged as its
    primitive part by convention.
    """
    form = request.form
    coefficients = tuple(
        value.as_fraction().numerator for value in form.diagonal_coefficients
    ) + tuple(term.coefficient.as_fraction().numerator for term in form.cross_terms)
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    if content == 0:
        primitive = form
    else:
        primitive = RationalQuadraticForm(
            axis=form.axis,
            diagonal_coefficients=tuple(
                CanonicalRational.from_integer_ratio(value // content, 1)
                for value in coefficients[: len(form.axis)]
            ),
            cross_terms=tuple(
                QuadraticCrossTerm(
                    left=term.left,
                    right=term.right,
                    coefficient=CanonicalRational.from_integer_ratio(
                        term.coefficient.as_fraction().numerator // content, 1
                    ),
                )
                for term in form.cross_terms
            ),
        )
    return IntegralContentResult.model_construct(
        form=form, content=content, primitive_part=primitive
    )


__all__ = [
    "bilinear_pairing",
    "coefficient_matrix",
    "coefficient_matrix_entries",
    "evaluate_rational_quadratic_form",
    "integral_coefficient_content",
    "require_coefficient_matrix_budget",
]
