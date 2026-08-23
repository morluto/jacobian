"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

MAX_ROWS = 256
MAX_COLUMNS = 256
MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


class PrimeFieldMatrixRequest(StrictModel):
    """A bounded integer matrix over an explicit prime field GF(p).

    The matrix is carried as the domain-owned ``PrimeFieldMatrix`` canonical
    value so it composes unchanged with the other GF(p) producers and
    consumers. Shape rules (schema-visible): 0..256 rows, 1..256 columns,
    rectangular rows, and every entry a canonical residue in ``[0, prime)``.
    Zero rows carry an explicit column axis, matching the canonical empty
    matrix that full-rank nullspace producers return. The characteristic is
    bounded to ``MAX_PRIME`` by a pre-construction validator so no accepted
    request can reach unbounded primality or modular work.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A bounded integer matrix over an explicit prime field GF(p), "
                "as the canonical `PrimeFieldMatrix` value: `prime` must be a "
                "prime integer in [2, 2147483647], `entries` must have at "
                "most 256 rows, each row 1..256 columns, all rows the same "
                "length matching the declared `columns`, and every entry a "
                "canonical residue in [0, prime). Zero rows are permitted "
                "and carry the explicit `columns` axis."
            )
        }
    )

    matrix: PrimeFieldMatrix

    @model_validator(mode="before")
    @classmethod
    def require_bounded_prime(cls, data: Any) -> Any:
        # Bound the characteristic BEFORE the nested canonical value is
        # constructed: PrimeFieldMatrix.__post_init__ runs the (expensive)
        # primality test, so an oversized prime must be rejected first.
        # Running a before validator moves field validation into Python
        # mode, where decoded JSON arrays no longer coerce to the declared
        # tuple shapes; normalize them here so JSON invocation keeps
        # working while every stored value stays a canonical tuple.
        if isinstance(data, dict):
            raw = data.get("matrix")
            if isinstance(raw, dict):
                prime = raw.get("prime")
                entries = raw.get("entries")
                if isinstance(entries, list):
                    # Copy along the rewritten path: the caller owns the
                    # payload, so normalization must not mutate it.
                    matrix = dict(raw)
                    matrix["entries"] = tuple(
                        tuple(row) if isinstance(row, list) else row for row in entries
                    )
                    data = {**data, "matrix": matrix}
            else:
                prime = getattr(raw, "prime", None)
            if isinstance(prime, int) and not 2 <= prime <= MAX_PRIME:
                raise ValueError(
                    "field prime must lie in [2, "
                    f"{MAX_PRIME}] so validation work stays bounded"
                )
        return data

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.matrix.entries) > MAX_ROWS:
            raise ValueError(f"matrix has at most {MAX_ROWS} rows")
        if not 1 <= self.matrix.columns <= MAX_COLUMNS:
            raise ValueError(f"matrix has at most {MAX_COLUMNS} columns")
        return self


def _require_source_prime(self_prime: int, source: PrimeFieldMatrixRequest) -> None:
    if self_prime != source.matrix.prime:
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
        if self.rank != kernel_rank(self.source.matrix):
            raise ValueError("rank does not match a recomputation from the source")
        return self


class PrimeFieldRrefResult(StrictModel):
    """Reduced row-echelon form over GF(p) as the canonical matrix value.

    ``rref_matrix`` carries the exact reduced form with its pivot columns and
    rank; it feeds downstream GF(p) consumers unchanged and is replayed
    against the retained source at validation.
    """

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rref_matrix: PrimeFieldMatrix
    pivot_columns: tuple[int, ...]
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_rref_canonical(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import rref as kernel_rref

        _require_source_prime(self.prime, self.source)
        expected_rows, expected_pivots = kernel_rref(self.source.matrix)
        if self.rref_matrix.entries != tuple(expected_rows):
            raise ValueError(
                "rref_matrix does not match a recomputation from the source"
            )
        if (
            self.rref_matrix.prime != self.source.matrix.prime
            or self.rref_matrix.columns != self.source.matrix.columns
        ):
            raise ValueError("rref_matrix must carry the source prime and column axis")
        if tuple(self.pivot_columns) != tuple(expected_pivots):
            raise ValueError("pivot_columns must be the exact pivot column sequence")
        if self.rank != len(expected_pivots):
            raise ValueError("rank must equal the number of pivot columns")
        return self


class PrimeFieldNullspaceResult(StrictModel):
    """Right nullspace basis over GF(p) as the canonical matrix value.

    An empty basis still carries the source ``columns`` axis inside
    ``nullspace_matrix`` so the ambient dimension of the zero subspace stays
    unambiguous and composable.
    """

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    nullspace_matrix: PrimeFieldMatrix
    nullity: int = Field(ge=0)

    @model_validator(mode="after")
    def require_nullspace_canonical(self) -> Self:
        from jacobian.math.prime_field_linear_algebra import (
            nullspace as kernel_nullspace,
        )

        _require_source_prime(self.prime, self.source)
        # Replay the conclusion from the retained source: the declared basis
        # and nullity must equal a full recomputation, which covers the
        # defining relation, independence, and completeness of the basis.
        expected_basis = kernel_nullspace(self.source.matrix)
        if self.nullspace_matrix.entries != tuple(expected_basis):
            raise ValueError(
                "nullspace_matrix does not match a recomputation from the source"
            )
        if (
            self.nullspace_matrix.prime != self.source.matrix.prime
            or self.nullspace_matrix.columns != self.source.matrix.columns
        ):
            raise ValueError(
                "nullspace_matrix must carry the source prime and column axis"
            )
        if self.nullity != len(expected_basis):
            raise ValueError("nullity must equal the number of basis vectors")
        return self


__all__ = [
    "MAX_COLUMNS",
    "MAX_PRIME",
    "MAX_ROWS",
    "PrimeFieldMatrixRankResult",
    "PrimeFieldMatrixRequest",
    "PrimeFieldNullspaceResult",
    "PrimeFieldRrefResult",
]
