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
MAX_PRIME = 1000003  # conservative prime modulus bound

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _require_bounded_prime(prime: int) -> None:
    if prime > MAX_PRIME:
        raise ValueError(f"prime {prime} exceeds the bounded modulus {MAX_PRIME} for field arithmetic")
    if not _is_prime(prime):
        raise ValueError(f"prime {prime} is not prime")


class PrimeFieldMatrixRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime(self.prime)
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
        _PrimeFieldMatrixValidator(prime=self.prime, entries=self.entries, columns=self.columns)
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
        _require_bounded_prime(self.prime)
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
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        _require_bounded_prime(self.prime)
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        if len(self.entries) > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        if any(len(row) != self.columns for row in self.entries):
            raise ValueError("every row must match the declared column count")
        if any(type(v) is not int or not 0 <= v < self.prime for row in self.entries for v in row):
            raise ValueError("entries must be canonical prime-field residues")
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, rref
        mat = PrimeFieldMatrix(prime=self.prime, entries=self.entries, columns=self.columns)
        _, pivots = rref(mat)
        expected = len(pivots)
        if self.rank != expected:
            raise ValueError(f"rank {self.rank} does not match recomputed rank {expected}")
        return self


class RrefRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime(self.prime)
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
    rref_matrix: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        if (
            self.rref_matrix.prime != self.prime
            or self.rref_matrix.columns != self.columns
        ):
            raise ValueError(
                "rref_matrix must carry the source prime and column count"
            )
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected_rows, expected_pivots = rref(matrix)
        if self.rref_matrix.entries != expected_rows:
            raise ValueError("rref_matrix must be the exact reduced row-echelon form")
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        return self


class NullspaceRequest(StrictModel):
    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime(self.prime)
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
    """The exact right-nullspace basis of the retained matrix."""

    nullspace_basis: PrimeFieldMatrix
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_NULLSPACE"] = "EXACT_DOMAIN_MATRIX_NULLSPACE"

    @model_validator(mode="after")
    def bind_nullspace(self) -> Self:
        if (
            self.nullspace_basis.prime != self.prime
            or self.nullspace_basis.columns != self.columns
        ):
            raise ValueError(
                "nullspace_basis must carry the source prime and column count"
            )
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected_rows = nullspace(matrix)
        expected = PrimeFieldMatrix(
            prime=self.prime, entries=expected_rows, columns=self.columns
        )
        if self.nullspace_basis != expected:
            raise ValueError(
                "nullspace_basis must be the exact nullspace basis of the "
                "retained matrix"
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
