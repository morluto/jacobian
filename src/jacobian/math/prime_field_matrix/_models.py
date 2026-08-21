"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ROWS = 256
MAX_COLUMNS = 256


class PrimeFieldMatrixRequest(StrictModel):
    """A bounded integer matrix over an explicit prime field GF(p)."""

    prime: int = Field(gt=1)
    entries: tuple[tuple[int, ...], ...] = Field(max_length=MAX_ROWS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = len(self.entries)
        if n == 0:
            raise ValueError("entries must be non-empty")
        columns = len(self.entries[0])
        if columns == 0:
            raise ValueError("matrix rows must be non-empty")
        if columns > MAX_COLUMNS:
            raise ValueError(f"matrix has at most {MAX_COLUMNS} columns")
        for row in self.entries:
            if len(row) != columns:
                raise ValueError("matrix rows must have the same column count")
        for row in self.entries:
            for value in row:
                if type(value) is not int or not 0 <= value < self.prime:
                    raise ValueError(
                        "matrix entries must be canonical residues in [0, prime)"
                    )
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        return self


class PrimeFieldMatrixRankResult(StrictModel):
    """Rank of a matrix over GF(p)."""

    prime: int = Field(gt=1)
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.rank > min(self.rows, self.columns):
            raise ValueError("rank cannot exceed min(rows, columns)")
        return self


class PrimeFieldRrefResult(StrictModel):
    """Reduced row-echelon form and pivot columns over GF(p)."""

    prime: int = Field(gt=1)
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    rref: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.rank != len(self.pivot_columns):
            raise ValueError("rank must equal the number of pivot columns")
        if len(self.rref) != self.rows:
            raise ValueError("rref must have the same row count as the input matrix")
        for row in self.rref:
            if len(row) != self.columns:
                raise ValueError("rref row length must match the input matrix")
        for col in self.pivot_columns:
            if not 0 <= col < self.columns:
                raise ValueError("pivot column index out of range")
        for row in self.rref:
            for value in row:
                if type(value) is not int or not 0 <= value < self.prime:
                    raise ValueError(
                        "rref entries must be canonical residues in [0, prime)"
                    )
        return self


class PrimeFieldNullspaceResult(StrictModel):
    """Right nullspace basis over GF(p)."""

    prime: int = Field(gt=1)
    columns: int = Field(ge=0)
    nullspace: tuple[tuple[int, ...], ...]
    nullity: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.nullity != len(self.nullspace):
            raise ValueError("nullity must equal the number of basis vectors")
        for row in self.nullspace:
            if len(row) != self.columns:
                raise ValueError("nullspace row length must match matrix columns")
            for value in row:
                if type(value) is not int or not 0 <= value < self.prime:
                    raise ValueError(
                        "nullspace entries must be canonical residues in [0, prime)"
                    )
        return self


__all__ = [
    "PrimeFieldMatrixRequest",
    "PrimeFieldMatrixRankResult",
    "PrimeFieldRrefResult",
    "PrimeFieldNullspaceResult",
]
