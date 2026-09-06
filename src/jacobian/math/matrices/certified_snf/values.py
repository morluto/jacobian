"""Semantic values for transformation-certified Smith normal forms."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.values import IntegerMatrix

MAX_CERTIFIED_SNF_DIMENSION = 32
MAX_CERTIFIED_SNF_INPUT_DIMENSION = 16
MAX_CERTIFIED_SNF_INPUT_DIGITS = 32
MAX_CERTIFIED_SNF_OUTPUT_DIGITS = 32_768


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


class SmithNormalFormCertificate(StrictModel):
    """A proposed exact relation ``D = U A V`` with unimodular ``U`` and ``V``."""

    source: IntegerMatrix
    diagonal: IntegerMatrix
    left_transformation: IntegerMatrix
    right_transformation: IntegerMatrix
    rank: StrictInt = Field(ge=0, le=MAX_CERTIFIED_SNF_DIMENSION)
    invariant_factors: tuple[ExactInteger, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_DIMENSION
    )
    left_determinant: ExactInteger = Field(ge=-1, le=1)
    right_determinant: ExactInteger = Field(ge=-1, le=1)
    relation: Literal["DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT"] = (
        "DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT"
    )
    transformation_scope: Literal["FULL_BASIS_BOTH_SIDES"] = "FULL_BASIS_BOTH_SIDES"
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

    @model_validator(mode="after")
    def require_unimodular_determinants(self) -> Self:
        if abs(self.left_determinant) != 1 or abs(self.right_determinant) != 1:
            raise _validation_error(
                "invalid",
                "Smith transformations must have determinant plus or minus one",
            )
        return self

    @model_validator(mode="after")
    def require_coherent_shapes_and_canonical_diagonal(self) -> Self:
        rows = self.source.row_count
        columns = self.source.column_count
        if (
            (self.diagonal.row_count, self.diagonal.column_count) != (rows, columns)
            or (
                self.left_transformation.row_count,
                self.left_transformation.column_count,
            )
            != (rows, rows)
            or (
                self.right_transformation.row_count,
                self.right_transformation.column_count,
            )
            != (columns, columns)
        ):
            raise _validation_error(
                "invalid", "Smith certificate matrix shapes are incompatible"
            )
        diagonal_count = min(rows, columns)
        diagonal = tuple(
            self.diagonal.entries[index][index] for index in range(diagonal_count)
        )
        if any(
            self.diagonal.entries[row][column] != 0
            for row in range(rows)
            for column in range(columns)
            if row != column
        ):
            raise _validation_error("invalid", "Smith normal form must be diagonal")
        factors = tuple(value for value in self.invariant_factors)
        if (
            self.rank != len(factors)
            or self.rank > diagonal_count
            or any(value <= 0 for value in factors)
            or diagonal[: self.rank] != factors
            or any(value != 0 for value in diagonal[self.rank :])
            or any(right % left for left, right in pairwise(factors))
        ):
            raise _validation_error(
                "invalid",
                "Smith invariant factors must be the positive divisibility diagonal",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: IntegerMatrix,
        diagonal: IntegerMatrix,
        left_transformation: IntegerMatrix,
        right_transformation: IntegerMatrix,
        rank: int,
        invariant_factors: tuple[ExactInteger, ...],
        left_determinant: int,
        right_determinant: int,
    ) -> Self:
        """Construct a certificate emitted by the trusted Smith kernel.

        Relation replay belongs to the explicit bounded verifier, not to
        deserialization of this canonical value.  The kernel has already
        established the relation before it reaches this factory.
        """

        return cls.model_construct(
            source=source,
            diagonal=diagonal,
            left_transformation=left_transformation,
            right_transformation=right_transformation,
            rank=rank,
            invariant_factors=invariant_factors,
            left_determinant=left_determinant,
            right_determinant=right_determinant,
        )


__all__ = ["IntegerMatrix", "SmithNormalFormCertificate"]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
