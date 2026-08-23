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
MAX_PRIME = 1000000


def _require_bounded_prime(prime: int) -> None:
    if prime > MAX_PRIME:
        raise ValueError("prime must be a prime modulus within the bounded domain")


class _PrimeFieldMatrixRequest(StrictModel):
    """One bounded matrix over an explicit bounded prime field.

    ``PrimeFieldMatrix`` is the domain-owned canonical value: it carries the
    prime, canonical residues, the declared column axis for empty matrices,
    and the dimension bound, so every request and result reuses it unchanged.
    """

    matrix: PrimeFieldMatrix

    @model_validator(mode="after")
    def require_bounded_prime(self) -> Self:
        _require_bounded_prime(self.matrix.prime)
        return self


class RankRequest(_PrimeFieldMatrixRequest):
    pass


class RankResult(RankRequest):
    rank: int = Field(ge=0)
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RANK"] = "EXACT_DOMAIN_MATRIX_RANK"

    @model_validator(mode="after")
    def bind_rank(self) -> Self:
        if self.rank > MAX_DIMENSION:
            raise ValueError("rank exceeds the supported dimension bound")
        _, expected_pivots = rref(self.matrix)
        if self.rank != len(expected_pivots):
            raise ValueError("rank does not match recomputation from the source matrix")
        return self


class RrefRequest(_PrimeFieldMatrixRequest):
    pass


class RrefResult(RrefRequest):
    rref: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_RREF"] = "EXACT_DOMAIN_MATRIX_RREF"

    @model_validator(mode="after")
    def bind_rref(self) -> Self:
        expected_rows, expected_pivots = rref(self.matrix)
        expected = PrimeFieldMatrix(
            prime=self.matrix.prime,
            entries=expected_rows,
            columns=self.matrix.columns,
        )
        if self.rref != expected:
            raise ValueError(
                "rref must be the exact reduced row-echelon form of the source matrix"
            )
        if self.pivot_columns != expected_pivots:
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        return self


class NullspaceRequest(_PrimeFieldMatrixRequest):
    pass


class NullspaceResult(NullspaceRequest):
    """The nullspace basis as the domain-owned canonical matrix value.

    ``nullspace_matrix`` retains the source prime and declared column axis,
    so an empty basis still names its ambient space and the serialized form
    feeds rank/RREF consumers unchanged.
    """

    nullspace_matrix: PrimeFieldMatrix
    complete: Literal[True] = True
    method: Literal["EXACT_DOMAIN_MATRIX_NULLSPACE"] = "EXACT_DOMAIN_MATRIX_NULLSPACE"

    @model_validator(mode="after")
    def bind_nullspace(self) -> Self:
        expected = nullspace(self.matrix)
        if self.nullspace_matrix.entries != tuple(expected):
            raise ValueError("nullspace_matrix must be the exact nullspace basis")
        if (
            self.nullspace_matrix.prime != self.matrix.prime
            or self.nullspace_matrix.columns != self.matrix.columns
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
