"""Typed wire contracts for cluster algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_EXCHANGE_SIZE = 16


def _require_shape(matrix: ExchangeMatrix) -> None:
    if len(matrix.entries) != matrix.n:
        raise ValueError("entries must be an n x n matrix")
    for row in matrix.entries:
        if len(row) != matrix.n:
            raise ValueError("entries must be a square matrix")
    if len(matrix.symmetrizer) != matrix.n:
        raise ValueError("symmetrizer must have n entries")


class ExchangeMatrix(StrictModel):
    """A skew-symmetrizable integer exchange matrix B.

    The symmetrizer D must have strictly positive diagonal entries: a
    diagonal matrix with positive diagonal satisfying DB = -B^T is exactly
    what makes B an exchange matrix, and a zero or negative entry would
    accept matrices that are not skew-symmetrizable.
    """

    n: int = Field(ge=1, le=MAX_EXCHANGE_SIZE)
    entries: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_EXCHANGE_SIZE)
    symmetrizer: tuple[int, ...] = Field(min_length=1, max_length=MAX_EXCHANGE_SIZE)

    @model_validator(mode="after")
    def require_valid_matrix(self) -> Self:
        _require_shape(self)
        for i in range(self.n):
            if self.symmetrizer[i] <= 0:
                raise ValueError(
                    "symmetrizer entries must be strictly positive integers"
                )
        for i in range(self.n):
            if self.entries[i][i] != 0:
                raise ValueError("diagonal entries must be zero")
        for i in range(self.n):
            for j in range(self.n):
                if self.symmetrizer[i] * self.entries[i][j] != -self.symmetrizer[j] * self.entries[j][i]:
                    raise ValueError(
                        f"skew-symmetrizability condition violated at ({i}, {j})"
                    )
        return self


class SeedMutationRequest(StrictModel):
    """Mutate a cluster seed at a specified mutable index."""

    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.mutation_index >= self.exchange_matrix.n:
            raise ValueError("mutation_index must be in 0..n-1")
        return self


class SeedMutationResult(StrictModel):
    """The mutated exchange matrix after applying the Fomin-Zelevinsky mutation."""

    exchange_matrix: ExchangeMatrix
    mutation_index: int = Field(ge=0)


class GVectorRequest(StrictModel):
    """Compute the g-vector matrix for principal coefficients."""

    exchange_matrix: ExchangeMatrix


class GVectorResult(StrictModel):
    """The g-vector matrix (identity for the initial seed)."""

    n: int = Field(ge=1)
    g_matrix: tuple[tuple[int, ...], ...]
    convention: str = "FOMIN_ZELEVINSKY"


__all__ = [
    "ExchangeMatrix",
    "GVectorRequest",
    "GVectorResult",
    "SeedMutationRequest",
    "SeedMutationResult",
]
