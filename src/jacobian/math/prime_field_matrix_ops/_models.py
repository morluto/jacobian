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
# The modulus carries at most MAX_MODULUS_DIGITS decimal digits (about 213
# bits): dense elimination performs at most MAX_DIMENSION**3 modular
# multiply-adds on residues below the modulus, so primality testing,
# inversion, Gaussian elimination, and result replay stay inside the declared
# work budget. Entries are canonical residues 0..prime-1 and therefore inherit
# the same ceiling.
MAX_MODULUS_DIGITS = 64
_MAX_MODULUS_MAGNITUDE = 10**MAX_MODULUS_DIGITS


def _require_bounded_prime_field_matrix(
    *, prime: int, entries: tuple[tuple[int, ...], ...], columns: int
) -> None:
    if prime >= _MAX_MODULUS_MAGNITUDE:
        raise ValueError(f"prime exceeds the {MAX_MODULUS_DIGITS}-digit modulus bound")
    if len(entries) > MAX_DIMENSION:
        raise ValueError("matrix exceeds the supported dimension bound")
    if any(len(row) != columns for row in entries):
        raise ValueError("every row must match the declared column count")
    if any(
        type(value) is not int or not 0 <= value < prime
        for row in entries
        for value in row
    ):
        raise ValueError("entries must be canonical prime-field residues")
    PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)


class PrimeFieldMatrixRequest(StrictModel):
    prime: int = Field(ge=2)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime_field_matrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        return self


class RankRequest(StrictModel):
    prime: int = Field(ge=2)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime_field_matrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        return self


class RankResult(RankRequest):
    """The exact rank bound to its source matrix.

    The canonical source matrix is retained and the rank is replayed
    against it during result validation, mirroring the RREF and nullspace
    results, so an authored rank cannot validate independently of the
    matrix it claims to describe.
    """

    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        matrix = PrimeFieldMatrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
        expected = rank(matrix)
        if self.rank != expected:
            raise ValueError("rank must be the exact rank of the source matrix")
        return self


class RrefRequest(StrictModel):
    prime: int = Field(ge=2)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime_field_matrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
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
    prime: int = Field(ge=2)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=0)
    columns: int = Field(ge=0, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_bounded_prime_field_matrix(
            prime=self.prime, entries=self.entries, columns=self.columns
        )
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
    "MAX_MODULUS_DIGITS",
    "NullspaceRequest",
    "NullspaceResult",
    "RankRequest",
    "RankResult",
    "RrefRequest",
    "RrefResult",
]
