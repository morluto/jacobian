"""Typed wire contracts for prime-field matrix operations."""

from __future__ import annotations

from typing import Self

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


class PrimeFieldMatrixRankResult(StrictModel):
    """Rank of a matrix over GF(p), bound to its source matrix."""

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        rows = len(self.source.entries)
        columns = len(self.source.entries[0]) if self.source.entries else 0
        if self.rank > min(rows, columns):
            raise ValueError("rank cannot exceed min(rows, columns)")
        return self


class PrimeFieldRrefResult(StrictModel):
    """Reduced row-echelon form and pivot columns over GF(p), bound to source."""

    prime: int = Field(gt=1, le=MAX_PRIME)
    source: PrimeFieldMatrixRequest
    rref: tuple[tuple[int, ...], ...]
    pivot_columns: tuple[int, ...]
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_rref_canonical(self) -> Self:  # noqa: C901
        rows = len(self.source.entries)
        columns = len(self.source.entries[0]) if self.source.entries else 0
        if self.rank != len(self.pivot_columns):
            raise ValueError("rank must equal the number of pivot columns")
        if len(self.rref) != rows:
            raise ValueError("rref must have the same row count as the input matrix")
        for row in self.rref:
            if len(row) != columns:
                raise ValueError("rref row length must match the input matrix")
        for col in self.pivot_columns:
            if not 0 <= col < columns:
                raise ValueError("pivot column index out of range")
        # Replay the RREF defining invariants.
        pivot_set = set(self.pivot_columns)
        leading: list[int] = []
        for row_index, row in enumerate(self.rref):
            pivots_in_row = [c for c, value in enumerate(row) if value != 0]
            is_zero_row = not pivots_in_row
            if row_index < len(self.pivot_columns):
                expected_col = self.pivot_columns[row_index]
                if is_zero_row or row[expected_col] != 1:
                    raise ValueError(
                        "each pivot row must have a leading one at its pivot column"
                    )
                if any(
                    value != 0
                    for c, value in enumerate(row)
                    if c in pivot_set and c != expected_col
                ):
                    raise ValueError(
                        "pivot columns must be cleared in all other pivot rows"
                    )
                leading.append(expected_col)
            elif not is_zero_row:
                # Non-pivot rows must be zero rows appearing after pivots.
                raise ValueError("non-pivot rows must be zero rows")
        if leading != sorted(leading):
            raise ValueError("pivot columns must strictly increase down the matrix")
        for row in self.rref:
            for value in row:
                if type(value) is not int or not 0 <= value < self.prime:
                    raise ValueError(
                        "rref entries must be canonical residues in [0, prime)"
                    )
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
    def require_nullspace_canonical(self) -> Self:  # noqa: C901
        columns = len(self.source.entries[0]) if self.source.entries else 0
        if self.nullity != len(self.nullspace):
            raise ValueError("nullity must equal the number of basis vectors")
        for basis_vector in self.nullspace:
            if len(basis_vector) != columns:
                raise ValueError("nullspace row length must match matrix columns")
            for value in basis_vector:
                if type(value) is not int or not 0 <= value < self.prime:
                    raise ValueError(
                        "nullspace entries must be canonical residues in [0, prime)"
                    )
        # Replay independence and the defining relation M @ v == 0.
        basis: list[tuple[int, ...]] = []
        for basis_vector in self.nullspace:
            if not any(basis_vector):
                raise ValueError("nullspace basis vectors must be nonzero")
            for row in self.source.entries:
                total = sum(a * b for a, b in zip(row, basis_vector, strict=True))
                if total % self.prime != 0:
                    raise ValueError(
                        "nullspace vectors must satisfy the defining relation"
                    )
            basis.append(tuple(basis_vector))
        # Independence: the only rational combination equal to zero is trivial.
        # With at most MAX_COLUMNS vectors of length <= MAX_COLUMNS this
        # bounded Gaussian elimination stays inside the request domain.
        rank = 0
        temp: list[list[int]] = [list(v) for v in basis]
        col_count = len(basis[0]) if basis else 0
        pivot_row = 0
        for col in range(col_count):
            pivot = next(
                (r for r in range(pivot_row, len(temp)) if temp[r][col] != 0), None
            )
            if pivot is None:
                continue
            temp[pivot_row], temp[pivot] = temp[pivot], temp[pivot_row]
            lead_inv = pow(temp[pivot_row][col], -1, self.prime)
            temp[pivot_row] = [
                (value * lead_inv) % self.prime for value in temp[pivot_row]
            ]
            for r in range(len(temp)):
                if r != pivot_row and temp[r][col] != 0:
                    factor = temp[r][col]
                    temp[r] = [
                        (a - factor * b) % self.prime
                        for a, b in zip(temp[r], temp[pivot_row], strict=True)
                    ]
            pivot_row += 1
            rank += 1
        if rank != len(basis):
            raise ValueError("nullspace basis vectors must be independent")
        return self


__all__ = [
    "PrimeFieldMatrixRankResult",
    "PrimeFieldMatrixRequest",
    "PrimeFieldNullspaceResult",
    "PrimeFieldRrefResult",
]
