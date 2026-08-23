"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel

MAX_ROWS = 256
MAX_COLUMNS = 256
MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


class PrimeFieldMatrixRequest(StrictModel):
    """A bounded integer matrix over an explicit prime field GF(p).

    Shape rules (schema-visible): at least one row, 1..256 columns per row,
    rectangular rows, and every entry a canonical residue in ``[0, prime)``.
    The prime itself is bounded to ``MAX_PRIME`` before primality testing so
    validation work stays bounded.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded integer matrix over an explicit prime field GF(p). "
                "`prime` must be a prime integer in [2, 2147483647]. `entries` "
                "must have at least one row, each row 1..256 columns, all rows "
                "the same length, and every entry a canonical residue in "
                "[0, prime)."
            )
        }
    )

    prime: int = Field(gt=1, le=MAX_PRIME)
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


def _kernel_matrix(source: PrimeFieldMatrixRequest) -> Any:
    from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

    return PrimeFieldMatrix(
        prime=source.prime,
        entries=source.entries,
        columns=len(source.entries[0]) if source.entries else 0,
    )


def _require_source_prime(self_prime: int, source: PrimeFieldMatrixRequest) -> None:
    if self_prime != source.prime:
        raise ValueError("result prime must equal the retained source prime")


class PrimeFieldMatrixRankResult(StrictModel):
    """Rank of a matrix over GF(p), bound to its source matrix."""

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import rank as kernel_rank

        _require_source_prime(self.prime, self.source)
        # Replay the conclusion from the retained source: the declared rank
        # must equal a recomputation, so corrupted results cannot revalidate.
        if self.rank != kernel_rank(_kernel_matrix(self.source)):
            raise ValueError("rank does not match a recomputation from the source")
        return self


class PrimeFieldRrefResult(StrictModel):
    """Reduced row-echelon form and pivot columns over GF(p), bound to source."""

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rref: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_rref_canonical(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import rref as kernel_rref

        _require_source_prime(self.prime, self.source)
        rows = len(self.source.entries)
        columns = len(self.source.entries[0]) if self.source.entries else 0
        if len(self.rref) != rows:
            raise ValueError("rref must have the same row count as the input matrix")
        for row in self.rref:
            if len(row) != columns:
                raise ValueError("rref row length must match the input matrix")
        # Replay the conclusion from the retained source: the declared RREF,
        # pivot columns, and rank must equal a full recomputation.
        expected_rows, expected_pivots = kernel_rref(_kernel_matrix(self.source))
        if (
            tuple(self.rref) != tuple(expected_rows)
            or tuple(self.pivot_columns) != tuple(expected_pivots)
            or self.rank != len(expected_pivots)
        ):
            raise ValueError("rref does not match a recomputation from the source")
        return self


class PrimeFieldNullspaceResult(StrictModel):
    """Right nullspace basis over GF(p), bound to its source matrix.

    An empty basis still carries ``columns`` so the ambient dimension of the
    zero subspace remains unambiguous.
    """

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    nullspace: tuple[tuple[int, ...], ...]
    nullity: int = Field(ge=0)

    @model_validator(mode="after")
    def require_nullspace_canonical(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import (
            nullspace as kernel_nullspace,
        )

        _require_source_prime(self.prime, self.source)
        columns = len(self.source.entries[0]) if self.source.entries else 0
        for basis_vector in self.nullspace:
            if len(basis_vector) != columns:
                raise ValueError("nullspace row length must match matrix columns")
        # Replay the conclusion from the retained source: the declared basis
        # and nullity must equal a full recomputation, which covers the
        # defining relation, independence, and completeness of the basis.
        expected_basis = kernel_nullspace(_kernel_matrix(self.source))
        if tuple(self.nullspace) != tuple(expected_basis):
            raise ValueError("nullspace does not match a recomputation from the source")
        if self.nullity != len(expected_basis):
            raise ValueError("nullity must equal the number of basis vectors")
        return self


__all__ = [
    "PrimeFieldMatrixRankResult",
    "PrimeFieldMatrixRequest",
    "PrimeFieldNullspaceResult",
    "PrimeFieldRrefResult",
]
