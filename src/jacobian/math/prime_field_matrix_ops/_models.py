"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)

MAX_DIMENSION = 256
# Explicit safe-number work bound: the field characteristic and every residue
# stay inside the strict interoperable JSON safe-integer range, so number-based
# JSON clients cannot silently round a request into a different matrix, and
# modular elimination work is bounded by machine-word-sized operands.
MAX_PRIME = 2**53 - 1


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

    def __init__(self, prime, entries, columns):
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


class RankResult(RankRequest):
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        if self.rank != rank(matrix):
            raise ValueError("rank must be the exact rank of the retained matrix")
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
    """The reduced row-echelon form as the domain-owned matrix value.

    ``reduced_matrix`` carries ``{prime, entries, columns}``, so it composes
    unchanged with ``prime_field_matrix.rank.compute`` and
    ``prime_field_matrix.nullspace.compute``; the retained ``columns`` axis
    keeps zero-row reductions well-formed matrix values.
    """

    reduced_matrix: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected_rows, expected_pivots = rref(matrix)
        expected = PrimeFieldMatrix(
            prime=self.prime, entries=expected_rows, columns=self.columns
        )
        if self.reduced_matrix != expected:
            raise ValueError(
                "reduced_matrix must be the exact reduced row-echelon form "
                "of the retained source matrix over the same prime"
            )
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
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
    """The nullspace basis as the domain-owned canonical matrix value.

    ``nullspace_matrix`` carries the source prime and the declared column
    axis, so an empty basis still names its ambient space and the serialized
    form feeds rank/RREF consumers unchanged.
    """

    nullspace_matrix: PrimeFieldMatrix
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_NULLSPACE"] = "EXACT_DOMAIN_MATRIX_NULLSPACE"

    @model_validator(mode="after")
    def bind_nullspace(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected = nullspace(matrix)
        if self.nullspace_matrix.entries != tuple(expected):
            raise ValueError("nullspace_matrix must be the exact nullspace basis")
        if (
            self.nullspace_matrix.prime != self.prime
            or self.nullspace_matrix.columns != self.columns
        ):
            raise ValueError(
                "nullspace_matrix must carry the source prime and column axis"
            )
        return self


__all__ = [
    "NullspaceRequest",
    "NullspaceResult",
    "RankRequest",
    "RankResult",
    "RrefRequest",
    "RrefResult",
]
