"""Wire values for closed rational matrix Lie spans."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import decimal_digit_width
from jacobian.math.lie_algebras._models import FiniteDimensionalLieAlgebra
from jacobian.math.matrices.values import RationalMatrix

MAX_MATRIX_SPAN_DIMENSION = 8
MAX_MATRIX_ORDER = 8
MAX_MATRIX_SPAN_INPUT_DIGITS = 64
MAX_MATRIX_SPAN_RESULT_DIGITS = 64
MAX_MATRIX_SPAN_OUTPUT_BYTES = 250_000


def _decimal_width(component: object) -> int:
    if isinstance(component, int):
        return decimal_digit_width(abs(component))
    return len(str(component).lstrip("-"))


class LieMatrixSpanRequest(StrictModel):
    """An independent ordered basis for a commutator-closed QQ matrix span."""

    matrices: tuple[RationalMatrix, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_matrices(cls, value: object) -> object:
        if not isinstance(value, dict) or not isinstance(
            value.get("matrices"), (list, tuple)
        ):
            return value
        matrices = value["matrices"]
        if not 1 <= len(matrices) <= MAX_MATRIX_SPAN_DIMENSION:
            raise PydanticCustomError(
                "lie_algebra.matrix_span_dimension",
                "matrix span dimension must be 1..8",
            )
        for matrix in matrices:
            if not isinstance(matrix, dict):
                continue
            entries = matrix.get("entries")
            if (
                not isinstance(entries, (list, tuple))
                or not 1 <= len(entries) <= MAX_MATRIX_ORDER
            ):
                raise PydanticCustomError(
                    "lie_algebra.matrix_span_order", "matrix order must be 1..8"
                )
            if any(
                not isinstance(row, (list, tuple)) or len(row) != len(entries)
                for row in entries
            ):
                raise PydanticCustomError(
                    "lie_algebra.matrix_span_shape", "each matrix must be square"
                )
            for row in entries:
                for scalar in row:
                    components = (
                        (scalar.get("num", "0"), scalar.get("den", "1"))
                        if isinstance(scalar, dict)
                        else (scalar,)
                    )
                    if any(
                        _decimal_width(component) > MAX_MATRIX_SPAN_INPUT_DIGITS
                        for component in components
                    ):
                        raise PydanticCustomError(
                            "lie_algebra.matrix_span_input_height",
                            "input matrix entries are limited to 64 decimal digits",
                        )
        return value


class LieMatrixSpanRealization(StrictModel):
    """A Lie algebra basis together with its exact ordered matrix realization."""

    algebra: FiniteDimensionalLieAlgebra
    matrix_basis: tuple[RationalMatrix, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_aligned_basis(self) -> Self:
        if len(self.matrix_basis) != len(self.algebra.basis):
            raise PydanticCustomError(
                "lie_algebra.matrix_span_axis",
                "matrix basis and Lie algebra basis axes must agree",
            )
        return self


__all__ = ["LieMatrixSpanRealization", "LieMatrixSpanRequest"]
