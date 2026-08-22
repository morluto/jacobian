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
        raise ValueError(
            f"prime {prime} exceeds the bounded modulus {MAX_PRIME} for field arithmetic"
        )
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
    """The exact rank of the retained prime-field matrix."""

    prime: int = Field(ge=2, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank_to_source(self) -> Self:
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        # Source-bound replay: the retained matrix must be a canonical
        # prime-field matrix whose exact rank equals the claimed value.
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, rank

        if len(self.entries) > MAX_DIMENSION or any(
            len(row) != self.columns for row in self.entries
        ):
            raise ValueError("retained matrix does not match its declared shape")
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected = rank(matrix)
        if self.rank != expected:
            raise ValueError(
                f"rank {self.rank} must be the exact rank {expected} of the "
                "retained matrix"
            )
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
    rref_rows: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    computed_matrix: PrimeFieldMatrix | None = None
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected_rows, expected_pivots = rref(matrix)
        # Canonical value: the computed matrix composes into downstream
        # prime-field consumers unchanged.
        if (
            self.computed_matrix is None
            or self.computed_matrix.prime != self.prime
            or self.computed_matrix.columns != self.columns
            or self.computed_matrix.entries != expected_rows
        ):
            raise ValueError(
                "computed_matrix must be the exact RREF over the same prime"
            )
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
    nullspace_rows: tuple[tuple[int, ...], ...]
    basis_matrix: PrimeFieldMatrix | None = None
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
        # Canonical value: the basis composes into downstream prime-field
        # consumers with its field context retained. A full-rank source
        # keeps the genuinely empty basis, matching nullspace_rows=().
        rows = tuple(tuple(int(v) for v in row) for row in self.nullspace_rows)
        if (
            self.basis_matrix is None
            or self.basis_matrix.prime != self.prime
            or self.basis_matrix.entries != rows
            or self.basis_matrix.columns != self.columns
        ):
            raise ValueError(
                "basis_matrix must be the exact nullspace basis over the "
                "same prime"
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
    "RankRequest",
    "RankResult",
    "RrefRequest",
    "RrefResult",
]
