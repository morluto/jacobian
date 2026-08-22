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


class _BoundedMatrixRequest(StrictModel):
    """One prime-field matrix within the operation's dimension budget."""

    matrix: PrimeFieldMatrix

    @model_validator(mode="after")
    def require_within_budget(self) -> Self:
        if len(self.matrix.entries) > MAX_DIMENSION or self.matrix.columns > MAX_DIMENSION:
            raise ValueError("matrix exceeds the supported dimension bound")
        return self


class RankRequest(_BoundedMatrixRequest):
    pass


class RankResult(RankRequest):
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import rank as pf_rank

        # Replay the defining rank invariant from the retained source matrix.
        if self.rank != pf_rank(self.matrix):
            raise ValueError("rank must be the exact prime-field matrix rank")
        if self.rank > min(len(self.matrix.entries), self.matrix.columns):
            raise ValueError("rank cannot exceed min(rows, columns)")
        return self


class RrefRequest(_BoundedMatrixRequest):
    pass


class RrefResult(RrefRequest):
    rref_rows: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        expected_rows, expected_pivots = rref(self.matrix)
        if self.rref_rows != expected_rows:
            raise ValueError("rref_rows must be the exact reduced row-echelon form")
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        return self


class NullspaceRequest(_BoundedMatrixRequest):
    pass


class NullspaceResult(NullspaceRequest):
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
