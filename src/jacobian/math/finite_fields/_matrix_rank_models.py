"""Typed contracts for finite-field matrix rank."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.finite_fields.values import AxisBoundMatrix


class MatrixRankRequest(StrictModel):
    """Compute the rank of one labelled matrix over its presented finite field."""

    matrix: AxisBoundMatrix

    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: object) -> object:
        return canonicalize_json_containers(data)


class MatrixRankResult(StrictModel):
    """Exact rank of a labelled matrix over its finite field."""

    matrix: AxisBoundMatrix
    rank: int = Field(ge=0)
    pivot_rows: tuple[str, ...] = Field(default=())
    pivot_columns: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_pivots(self) -> Self:
        if len(self.pivot_rows) != self.rank:
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_row_count",
                "pivot row count must equal rank",
            )
        if len(self.pivot_columns) != self.rank:
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_column_count",
                "pivot column count must equal rank",
            )
        row_labels = set(self.matrix.row_axis.labels)
        col_labels = set(self.matrix.column_axis.labels)
        if len(set(self.pivot_rows)) != len(self.pivot_rows):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_rows_unique",
                "pivot rows must be distinct",
            )
        if len(set(self.pivot_columns)) != len(self.pivot_columns):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_columns_unique",
                "pivot columns must be distinct",
            )
        if self.rank > min(len(row_labels), len(col_labels)):
            raise PydanticCustomError(
                "finite_field.matrix_rank.rank_within_dimensions",
                "rank must not exceed either matrix dimension",
            )
        if any(label not in row_labels for label in self.pivot_rows):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_row_labels",
                "pivot rows must be declared row labels",
            )
        row_positions = {
            label: position
            for position, label in enumerate(self.matrix.row_axis.labels)
        }
        if any(
            row_positions[later] <= row_positions[earlier]
            for earlier, later in zip(
                self.pivot_rows, self.pivot_rows[1:], strict=False
            )
        ):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_rows_order",
                "pivot rows must follow the declared row-axis order",
            )
        if any(label not in col_labels for label in self.pivot_columns):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_column_labels",
                "pivot columns must be declared column labels",
            )
        column_positions = {
            label: position
            for position, label in enumerate(self.matrix.column_axis.labels)
        }
        if any(
            column_positions[later] <= column_positions[earlier]
            for earlier, later in zip(
                self.pivot_columns, self.pivot_columns[1:], strict=False
            )
        ):
            raise PydanticCustomError(
                "finite_field.matrix_rank.pivot_columns_order",
                "pivot columns must follow the declared column-axis order",
            )
        return self


__all__ = [
    "MatrixRankRequest",
    "MatrixRankResult",
]
