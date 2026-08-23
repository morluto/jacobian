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


class RankRequest(StrictModel):
    """Rank of one bounded matrix over an explicit prime field.

    The matrix is the domain-owned canonical ``PrimeFieldMatrix`` value, so a
    matrix produced by any prime-field operation enters unchanged.
    """

    matrix: PrimeFieldMatrix


class RankResult(StrictModel):
    """The exact rank bound to the retained source matrix."""

    matrix: PrimeFieldMatrix
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank_to_source(self) -> Self:
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        # Source-bound replay: the retained canonical matrix's exact rank
        # must equal the claimed value.
        from jacobian.math.prime_field_linear_algebra import rank

        expected = rank(self.matrix)
        if self.rank != expected:
            raise ValueError(
                f"rank {self.rank} must be the exact rank {expected} of the "
                "retained matrix"
            )
        return self


class RrefRequest(StrictModel):
    """Reduced row-echelon form of one bounded matrix over a prime field.

    Accepts the domain-owned canonical ``PrimeFieldMatrix`` value so the
    transformed matrix returned by ``RrefResult`` re-enters unchanged.
    """

    matrix: PrimeFieldMatrix


class RrefResult(StrictModel):
    """The exact RREF as a canonical prime-field matrix value.

    ``rref`` is the domain-owned matrix value, so downstream rank and
    nullspace consumers accept it unchanged instead of extracting row tuples
    into flat request fields.
    """

    matrix: PrimeFieldMatrix
    rref: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        if self.rref.prime != self.matrix.prime or (
            self.rref.columns != self.matrix.columns
        ):
            raise ValueError(
                "rref must be a matrix value over the source ring with the "
                "source shape"
            )
        expected_rows, expected_pivots = rref(self.matrix)
        if self.rref.entries != expected_rows:
            raise ValueError("rref must be the exact reduced row-echelon form")
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        return self


class NullspaceRequest(StrictModel):
    """Right nullspace basis of one bounded matrix over a prime field.

    Accepts the domain-owned canonical ``PrimeFieldMatrix`` value, including
    the transformed matrix carried by an ``RrefResult``.
    """

    matrix: PrimeFieldMatrix


class NullspaceResult(StrictModel):
    """The deterministic nullspace basis bound to the retained source matrix."""

    matrix: PrimeFieldMatrix
    nullspace_rows: tuple[tuple[int, ...], ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_NULLSPACE"] = "EXACT_DOMAIN_MATRIX_NULLSPACE"

    @model_validator(mode="after")
    def bind_nullspace(self) -> Self:
        expected = nullspace(self.matrix)
        if self.nullspace_rows != expected:
            raise ValueError("nullspace_rows must be the exact nullspace basis")
        for vector in self.nullspace_rows:
            if len(vector) != self.matrix.columns:
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
