"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rref,
)

MAX_DIMENSION = 256


MAX_PRIME = 100000


class PrimeFieldMatrixValue(StrictModel):
    """The one canonical prime-field matrix value.

    Produced by RREF and other matrix operations and accepted unchanged by
    rank, RREF, and nullspace consumers, so a serialized RREF result composes
    without reconstructing prime/entries/columns in parallel.
    """

    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("entries must be canonical prime-field residues")
        PrimeFieldMatrix(prime=self.prime, entries=self.entries, columns=self.columns)
        return self


class PrimeFieldMatrixRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("entries must be canonical prime-field residues")
        if not self.entries and self.columns == 0:
            return self
        _PrimeFieldMatrixValidator(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        return self


class _PrimeFieldMatrixValidator:
    """Trigger PrimeFieldMatrix validation."""

    def __init__(
        self,
        prime: int,
        entries: tuple[tuple[int, ...], ...],
        columns: int,
    ) -> None:
        PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)


class RankRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("entries must be canonical prime-field residues")
        PrimeFieldMatrix(prime=self.prime, entries=self.entries, columns=self.columns)
        return self


class RankResult(StrictModel):
    """The exact rank of one retained source matrix over GF(prime).

    Carries the matrix so result validation replays the rank invariant
    instead of trusting an independently authored integer.
    """

    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)
    rank: int = Field(ge=0)
    prime: int = Field(ge=2, le=MAX_PRIME)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        if len(self.entries) > MAX_DIMENSION or any(
            len(row) != self.columns for row in self.entries
        ):
            raise ValueError("entries must match the declared shape")
        from jacobian.math.prime_field_linear_algebra import rank as _rank

        expected = _rank(
            PrimeFieldMatrix(
                prime=self.prime, entries=self.entries, columns=self.columns
            )
        )
        if self.rank != expected:
            raise ValueError("rank must be the exact rank of the source matrix")
        return self


class RrefRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("entries must be canonical prime-field residues")
        PrimeFieldMatrix(prime=self.prime, entries=self.entries, columns=self.columns)
        return self


class RrefResult(RrefRequest):
    rref_rows: tuple[tuple[int, ...], ...]
    rref: PrimeFieldMatrixValue
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected_rows, expected_pivots = rref(matrix)
        if self.rref_rows != expected_rows:
            raise ValueError("rref_rows must be the exact reduced row-echelon form")
        expected_rref = PrimeFieldMatrixValue(
            prime=self.prime, entries=expected_rows, columns=self.columns
        )
        if self.rref != expected_rref:
            raise ValueError(
                "rref must be the reusable prime-field matrix value of the RREF"
            )
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.rref_rows
            for value in row
        ):
            raise ValueError("rref entries must be canonical prime-field residues")
        return self


class NullspaceRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.entries
            for value in row
        ):
            raise ValueError("entries must be canonical prime-field residues")
        PrimeFieldMatrix(prime=self.prime, entries=self.entries, columns=self.columns)
        return self


class NullspaceResult(NullspaceRequest):
    nullspace_rows: tuple[tuple[int, ...], ...]
    nullspace: PrimeFieldMatrixValue
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_NULLSPACE"] = "EXACT_DOMAIN_MATRIX_NULLSPACE"

    @model_validator(mode="after")
    def bind_nullspace(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected = nullspace(matrix)
        if self.nullspace_rows != expected:
            raise ValueError("nullspace_rows must be the exact nullspace basis")
        expected_nullspace = PrimeFieldMatrixValue(
            prime=self.prime,
            entries=expected,
            columns=self.columns,
        )
        if self.nullspace != expected_nullspace:
            raise ValueError(
                "nullspace must be the reusable prime-field matrix value of the basis"
            )
        if any(
            type(value) is not int or not 0 <= value < self.prime
            for row in self.nullspace_rows
            for value in row
        ):
            raise ValueError("nullspace entries must be canonical prime-field residues")
        for vector in self.nullspace_rows:
            if len(vector) != self.columns:
                raise ValueError("nullspace vector length must match matrix columns")
        return self


__all__ = [
    "NullspaceRequest",
    "NullspaceResult",
    "PrimeFieldMatrixValue",
    "RankRequest",
    "RankResult",
    "RrefRequest",
    "RrefResult",
]
