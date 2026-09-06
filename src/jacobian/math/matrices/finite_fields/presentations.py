"""Native coordinate conversions; field projections are not catalog operations."""

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def bind_prime_matrix(
    matrix: PrimeFieldMatrix,
    presentation: FiniteFieldPresentation,
    row_axis: Axis,
    column_axis: Axis,
) -> AxisBoundMatrix:
    if presentation.degree != 1 or presentation.characteristic != matrix.prime:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.prime_presentation",
            message="presentation must have the matrix prime and degree one",
        )
    if (
        len(row_axis.labels) != len(matrix.entries)
        or len(column_axis.labels) != matrix.columns
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="finite_field.axis_shape",
            message="declared axes must match the positional matrix shape",
        )
    if len(matrix.entries) > 256 or matrix.columns > 256:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="finite_field.presentation_axis_bound",
            message="presented matrices admit axes through 256 labels",
        )
    require_field(presentation)
    return AxisBoundMatrix.model_construct(
        presentation=presentation,
        row_axis=row_axis,
        column_axis=column_axis,
        entries=tuple(
            tuple(
                FiniteFieldElement.model_construct(
                    presentation=presentation, coordinates=(value,)
                )
                for value in row
            )
            for row in matrix.entries
        ),
    )


def prime_matrix_coordinates(matrix: AxisBoundMatrix) -> PrimeFieldMatrix:
    """Return positional coordinates; the caller retains the labelled source."""
    if matrix.presentation.degree != 1:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="finite_field.prime_presentation",
            message="prime matrix coordinates require a degree-one presentation",
        )
    require_field(matrix.presentation)
    return PrimeFieldMatrix._from_admitted(
        prime=matrix.presentation.characteristic,
        columns=len(matrix.column_axis.labels),
        entries=tuple(
            tuple(value.coordinates[0] for value in row) for row in matrix.entries
        ),
    )


__all__ = ["bind_prime_matrix", "prime_matrix_coordinates"]
