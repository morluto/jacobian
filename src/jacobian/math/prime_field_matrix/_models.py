"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

MAX_ROWS = 256
MAX_COLUMNS = 256
MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


class PrimeFieldMatrixRequest(StrictModel):
    """A bounded integer matrix over an explicit prime field GF(p).

    Exactly one source shape is accepted: either the separate ``prime`` and
    ``entries`` fields, optionally restating ``columns``, or one canonical
    ``PrimeFieldMatrix`` value under ``matrix``, so a serialized GF(p)
    matrix composes unchanged with these consumers. Both shapes admit the
    same mathematical domain -- at least one row, 1..256 rectangular
    columns, canonical residues in ``[0, prime)``, and a prime
    characteristic at most ``MAX_PRIME``; this request owns that envelope
    for every accepted shape, so an already-canonical value cannot carry a
    larger characteristic past it.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded integer matrix over an explicit prime field GF(p). "
                "Provide either separate fields `prime` plus `entries` "
                "(optionally restating `columns`) or one canonical matrix "
                "value as `matrix` = {prime, entries, columns}, never both. "
                "`prime` must be a prime integer in [2, 2147483647]. "
                "`entries` must have at least one row, each row 1..256 "
                "columns, all rows the same length, and every entry a "
                "canonical residue in [0, prime)."
            )
        }
    )

    prime: int | None = Field(default=None, gt=1, le=MAX_PRIME)
    entries: tuple[tuple[int, ...], ...] | None = Field(
        default=None, max_length=MAX_ROWS
    )
    columns: int | None = Field(default=None, ge=0, le=MAX_COLUMNS)
    matrix: PrimeFieldMatrix | None = None

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.matrix is not None:
            _require_canonical_shape(self, self.matrix)
            return self
        _require_flat_shape(self)
        return self


def _require_canonical_shape(
    request: PrimeFieldMatrixRequest, value: PrimeFieldMatrix
) -> None:
    """Validate the embedded canonical-value shape at the request envelope."""

    if (
        request.prime is not None
        or request.entries is not None
        or request.columns is not None
    ):
        raise ValueError(
            "provide either the separate prime/entries fields or one "
            "canonical matrix value, not both"
        )
    # The request owns MAX_PRIME for every accepted shape.
    if value.prime > MAX_PRIME:
        raise ValueError(f"prime must be at most {MAX_PRIME}")
    # Keep one admitted domain across both shapes: the flat fields reject
    # empty matrices, so the embedded value must too.
    if not value.entries or value.columns == 0:
        raise ValueError("entries must be non-empty")


def _require_flat_shape(request: PrimeFieldMatrixRequest) -> None:
    """Validate the separate prime/entries(/columns) fields."""

    if request.prime is None or request.entries is None:
        raise ValueError(
            "either separate prime and entries or one canonical matrix "
            "value is required"
        )
    _require_entry_grid(request.entries, request.columns)
    _require_canonical_residues(request.entries, request.prime)
    from sympy import isprime

    if not isprime(request.prime):
        raise ValueError("prime must be a prime integer")


def _require_entry_grid(
    entries: tuple[tuple[int, ...], ...], declared_columns: int | None
) -> None:
    """Validate shape rules shared by both accepted source shapes."""

    n = len(entries)
    if n == 0:
        raise ValueError("entries must be non-empty")
    columns = len(entries[0])
    if columns == 0:
        raise ValueError("matrix rows must be non-empty")
    if columns > MAX_COLUMNS:
        raise ValueError(f"matrix has at most {MAX_COLUMNS} columns")
    if declared_columns is not None and declared_columns != columns:
        raise ValueError("columns must equal the shared entry row width")
    for row in entries:
        if len(row) != columns:
            raise ValueError("matrix rows must have the same column count")


def _require_canonical_residues(
    entries: tuple[tuple[int, ...], ...], prime: int
) -> None:
    """Every entry must be a canonical GF(prime) residue."""

    for row in entries:
        for value in row:
            if type(value) is not int or not 0 <= value < prime:
                raise ValueError(
                    "matrix entries must be canonical residues in [0, prime)"
                )


def _kernel_matrix(source: PrimeFieldMatrixRequest) -> PrimeFieldMatrix:
    """The one canonical kernel view of either accepted request shape."""

    if source.matrix is not None:
        return source.matrix
    prime = source.prime
    entries = source.entries
    if prime is None or entries is None:
        # Unreachable for a validated request; fail closed rather than
        # fabricate a matrix value.
        raise ValueError("request carries neither accepted source shape")
    return PrimeFieldMatrix(
        prime=prime,
        entries=entries,
        columns=len(entries[0]),
    )


def _require_source_prime(self_prime: int, source: PrimeFieldMatrixRequest) -> None:
    if self_prime != _kernel_matrix(source).prime:
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

        source_matrix = _kernel_matrix(self.source)
        _require_source_prime(self.prime, self.source)
        rows = len(source_matrix.entries)
        columns = source_matrix.columns
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
        columns = _kernel_matrix(self.source).columns
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
