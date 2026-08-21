"""Typed wire contracts for linear matroid operations over a finite field."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel

MAX_GROUND_SIZE = 32
MAX_ROWS = 16


def _require_prime(value: int) -> None:
    """Reject composite moduli: the kernel claims GF(p) Fermat inverses."""
    if value < 2 or any(
        value % divisor == 0
        for divisor in range(2, int(value**0.5) + 1)
    ):
        raise ValueError("prime must be a prime field modulus")


class LinearMatroid(StrictModel):
    """A matroid represented by columns of a matrix over a prime field.

    The ground set is {0, ..., n-1} where n is the number of columns.
    Each column is a vector over GF(p).
    """

    prime: int = Field(ge=2, le=10_000)
    num_rows: int = Field(ge=1, le=MAX_ROWS)
    columns: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_columns(self) -> Self:
        _require_prime(self.prime)
        if len(self.columns) < 1:
            raise ValueError("at least one column is required")
        for col in self.columns:
            if len(col) != self.num_rows:
                raise ValueError("each column must have num_rows entries")
            for entry in col:
                if not (0 <= entry < self.prime):
                    raise ValueError("entries must be in 0..p-1")
        return self


class MatroidRankRequest(StrictModel):
    """Compute the rank of a linear matroid."""

    matroid: LinearMatroid


class MatroidRankResult(StrictModel):
    """The rank of a matroid (dimension of its column space)."""

    rank: StrictInt = Field(ge=0)
    ground_size: StrictInt = Field(ge=1, le=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_rank(self) -> Self:
        if self.rank > min(self.ground_size, MAX_ROWS):
            raise ValueError("rank cannot exceed ground size or number of rows")
        return self


class MatroidClosureRequest(StrictModel):
    """Compute the closure of a subset in a linear matroid."""

    matroid: LinearMatroid
    subset: tuple[int, ...] = Field(min_length=0, max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        for idx in self.subset:
            if not (0 <= idx < len(self.matroid.columns)):
                raise ValueError("subset indices must be in 0..n-1")
        if len(set(self.subset)) != len(self.subset):
            raise ValueError("subset indices must be distinct")
        return self


class MatroidClosureResult(StrictModel):
    """The closure (flat) of a subset in a linear matroid."""

    closure: tuple[int, ...]
    closure_size: StrictInt = Field(ge=0, le=MAX_GROUND_SIZE)
    rank: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_closure(self) -> Self:
        if len(set(self.closure)) != len(self.closure):
            raise ValueError("closure elements must be distinct")
        if self.closure_size != len(self.closure):
            raise ValueError("closure_size must match closure length")
        return self


__all__ = [
    "LinearMatroid",
    "MatroidClosureRequest",
    "MatroidClosureResult",
    "MatroidRankRequest",
    "MatroidRankResult",
]
