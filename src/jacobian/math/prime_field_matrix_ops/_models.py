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
# Dense elimination and sympy primality testing stay cheap only for bounded
# moduli; residues are < prime, so this caps every residue's size too.
MAX_MATRIX_PRIME = 1_000_000_007


class PrimeFieldMatrixRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_MATRIX_PRIME)
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
    prime: int = Field(ge=2, le=MAX_MATRIX_PRIME)
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
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        # Replay the defining rank invariant from the retained source matrix.
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        from jacobian.math.prime_field_linear_algebra import rank as pf_rank

        expected = pf_rank(matrix)
        if self.rank != expected:
            raise ValueError("rank must be the exact prime-field matrix rank")
        # Ensure rank is consistent with dimensions.
        if self.rank > min(len(self.entries), self.columns):
            raise ValueError("rank cannot exceed min(rows, columns)")
        return self


class RrefRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_MATRIX_PRIME)
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
    prime: int = Field(ge=2, le=MAX_MATRIX_PRIME)
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
    "RankRequest",
    "RankResult",
    "RrefRequest",
    "RrefResult",
]
