"""Typed contract for integral unimodular quadratic-form changes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import Field, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import (
    IntegerMatrix,
    integer_matrix_axis_schema,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticForm,
)

MAX_UNIMODULAR_CHANGE_AXIS = 32


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.unimodular.{reason}", message)


class UnimodularChangeRequest(StrictModel):
    """A supplied integral basis map from target coordinates to source ones."""

    form: IntegralQuadraticForm
    matrix: Annotated[
        IntegerMatrix,
        WithJsonSchema(integer_matrix_axis_schema(MAX_UNIMODULAR_CHANGE_AXIS)),
    ] = Field(
        description=(
            "Square integer matrix M mapping target coordinates y to source "
            "coordinates x=M*y. At most 32 rows/columns and 128 decimal "
            "digits per entry are admitted; the operation requires det(M)=1 "
            "or -1."
        )
    )
    target_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_UNIMODULAR_CHANGE_AXIS,
        description="Ordered labels for the coordinates before the basis change.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_matrix_before_nested_parse(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw_matrix = value.get("matrix")
        if isinstance(raw_matrix, IntegerMatrix):
            row_count = raw_matrix.row_count
            column_count = raw_matrix.column_count
            entries = raw_matrix.entries
        elif isinstance(raw_matrix, Mapping):
            entries = raw_matrix.get("entries", ())
            if not isinstance(entries, (tuple, list)):
                return value
            row_count = raw_matrix.get("row_count", len(entries))
            column_count = raw_matrix.get(
                "column_count",
                len(entries[0])
                if entries and isinstance(entries[0], (tuple, list))
                else 0,
            )
        else:
            return value
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or not isinstance(column_count, int)
            or isinstance(column_count, bool)
            or row_count > MAX_UNIMODULAR_CHANGE_AXIS
            or column_count > MAX_UNIMODULAR_CHANGE_AXIS
        ):
            raise _error(
                "matrix_shape_bound", "change matrix is limited to 32 rows and columns"
            )
        if len(entries) > MAX_UNIMODULAR_CHANGE_AXIS:
            raise _error("matrix_shape_bound", "change matrix is limited to 32 rows")
        for row in entries:
            if not isinstance(row, (tuple, list)):
                return value
            if len(row) > MAX_UNIMODULAR_CHANGE_AXIS:
                raise _error(
                    "matrix_shape_bound", "change matrix is limited to 32 columns"
                )
            for entry in row:
                if isinstance(entry, str):
                    digits = len(entry.lstrip("-"))
                elif isinstance(entry, int) and not isinstance(entry, bool):
                    digits = 128 if abs(entry) < 10**128 else 129
                else:
                    continue
                if digits > 128:
                    raise _error(
                        "matrix_entry_bound",
                        "change matrix entries are limited to 128 digits",
                    )
        return value

    @model_validator(mode="after")
    def require_shape_and_axes(self) -> Self:
        dimension = len(self.form.axis)
        if dimension > MAX_UNIMODULAR_CHANGE_AXIS:
            raise _error("axis_bound", "unimodular change axis exceeds 32")
        if (
            self.matrix.row_count != dimension
            or self.matrix.column_count != dimension
            or len(self.target_axis) != dimension
        ):
            raise _error(
                "shape", "matrix and target axis must match the source dimension"
            )
        if len(set(self.target_axis)) != dimension:
            raise _error("axis_unique", "target coordinate labels must be unique")
        return self


class UnimodularChangeResult(StrictModel):
    """An integral presentation related by an exact invertible ZZ basis map."""

    source: IntegralQuadraticForm
    matrix: IntegerMatrix
    inverse: IntegerMatrix
    target: IntegralQuadraticForm

    @model_validator(mode="after")
    def require_transport_shapes(self) -> Self:
        dimension = len(self.source.axis)
        if (
            len(self.target.axis) != dimension
            or self.matrix.row_count != dimension
            or self.matrix.column_count != dimension
            or self.inverse.row_count != dimension
            or self.inverse.column_count != dimension
        ):
            raise _error("result_shape", "change maps and forms must share a dimension")
        if dimension > MAX_UNIMODULAR_CHANGE_AXIS:
            raise _error("result_shape", "unimodular results are limited to 32 coordinates")
        matrix = self.matrix.entries
        inverse = self.inverse.entries
        for left in range(dimension):
            for right in range(dimension):
                product = sum(matrix[left][index] * inverse[index][right] for index in range(dimension))
                reverse_product = sum(inverse[left][index] * matrix[index][right] for index in range(dimension))
                expected = int(left == right)
                if product != expected or reverse_product != expected:
                    raise _error("result_inverse", "inverse must be two-sided for the change matrix")

        source_cross = {
            (term.left, term.right): term.coefficient for term in self.source.cross_terms
        }
        target_diagonal: list[int] = []
        target_cross: list[tuple[int, int, int]] = []
        for left in range(dimension):
            diagonal = sum(
                coefficient * matrix[index][left] ** 2
                for index, coefficient in enumerate(self.source.diagonal_coefficients)
            )
            diagonal += sum(
                coefficient * matrix[first][left] * matrix[second][left]
                for (first, second), coefficient in source_cross.items()
            )
            target_diagonal.append(diagonal)
            for right in range(left + 1, dimension):
                cross = sum(
                    2 * coefficient * matrix[index][left] * matrix[index][right]
                    for index, coefficient in enumerate(self.source.diagonal_coefficients)
                )
                cross += sum(
                    coefficient
                    * (
                        matrix[first][left] * matrix[second][right]
                        + matrix[first][right] * matrix[second][left]
                    )
                    for (first, second), coefficient in source_cross.items()
                )
                if cross:
                    target_cross.append((left, right, cross))
        if (
            tuple(target_diagonal) != self.target.diagonal_coefficients
            or tuple(target_cross)
            != tuple(
                (term.left, term.right, term.coefficient)
                for term in self.target.cross_terms
            )
        ):
            raise _error("result_congruence", "target must equal the exact congruence image Q(M y)")
        return self


__all__ = [
    "MAX_UNIMODULAR_CHANGE_AXIS",
    "UnimodularChangeRequest",
    "UnimodularChangeResult",
]
