"""Semantic values for transformation-certified Smith normal forms."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

MAX_CERTIFIED_SNF_DIMENSION = 32
MAX_CERTIFIED_SNF_INPUT_DIMENSION = 16
MAX_CERTIFIED_SNF_INPUT_DIGITS = 32
MAX_CERTIFIED_SNF_OUTPUT_DIGITS = 32_768


def _integer_digits(value: str) -> int:
    return len(value.lstrip("-"))


class CertifiedIntegerMatrix(StrictModel):
    """One bounded integer matrix, including matrices with a zero dimension."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["ZZ"] = "ZZ"
    row_count: StrictInt = Field(ge=0, le=MAX_CERTIFIED_SNF_DIMENSION)
    column_count: StrictInt = Field(ge=0, le=MAX_CERTIFIED_SNF_DIMENSION)
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        default=(),
        max_length=MAX_CERTIFIED_SNF_DIMENSION,
    )

    @model_validator(mode="after")
    def require_declared_shape_and_output_budget(self) -> Self:
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise ValueError("certified integer matrix entries must match its shape")
        if any(
            _integer_digits(value) > MAX_CERTIFIED_SNF_OUTPUT_DIGITS
            for row in self.entries
            for value in row
        ):
            raise ValueError("certified integer matrix exceeds the output digit bound")
        return self


class SmithNormalFormCertificate(StrictModel):
    """A proposed exact relation ``D = U A V`` with unimodular ``U`` and ``V``."""

    certificate_schema_version: Literal["1"] = "1"
    source: CertifiedIntegerMatrix
    diagonal: CertifiedIntegerMatrix
    left_transformation: CertifiedIntegerMatrix
    right_transformation: CertifiedIntegerMatrix
    rank: StrictInt = Field(ge=0, le=MAX_CERTIFIED_SNF_DIMENSION)
    invariant_factors: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_CERTIFIED_SNF_DIMENSION
    )
    left_determinant: Literal["-1", "1"]
    right_determinant: Literal["-1", "1"]
    relation: Literal["DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT"] = (
        "DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT"
    )
    transformation_scope: Literal["FULL_BASIS_BOTH_SIDES"] = "FULL_BASIS_BOTH_SIDES"
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

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
            raise ValueError("Smith certificate matrix shapes are incompatible")
        diagonal_count = min(rows, columns)
        diagonal = tuple(
            parse_canonical_integer(self.diagonal.entries[index][index])
            for index in range(diagonal_count)
        )
        if any(
            parse_canonical_integer(self.diagonal.entries[row][column]) != 0
            for row in range(rows)
            for column in range(columns)
            if row != column
        ):
            raise ValueError("Smith normal form must be diagonal")
        factors = tuple(
            parse_canonical_integer(value) for value in self.invariant_factors
        )
        if (
            self.rank != len(factors)
            or self.rank > diagonal_count
            or any(value <= 0 for value in factors)
            or diagonal[: self.rank] != factors
            or any(value != 0 for value in diagonal[self.rank :])
            or any(right % left for left, right in pairwise(factors))
        ):
            raise ValueError(
                "Smith invariant factors must be the positive divisibility diagonal"
            )
        return self


__all__ = ["CertifiedIntegerMatrix", "SmithNormalFormCertificate"]
