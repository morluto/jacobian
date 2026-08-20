"""Provider-independent values for exact combinatorial-matrix operations.

A *sign matrix* is a rectangular matrix whose entries are structurally in
``{-1, +1}``.  A *Hadamard matrix* is a square sign matrix ``H`` satisfying
``H H^T = n I_n`` exactly.

Orthogonality is a construction invariant of the :class:`HadamardMatrix`
value.  Untrusted external JSON first produces a sign matrix/profile; only a
successful exact profile constructs a :class:`HadamardMatrix`.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_MATRIX_ORDER = 128


def _validate_sign_entries(rows: tuple[tuple[int, ...], ...]) -> None:
    for row in rows:
        for entry in row:
            if entry not in (-1, 1):
                raise ValueError("sign matrix entries must be -1 or +1")


class SignMatrix(StrictModel):
    """A bounded rectangular matrix whose entries are in ``{-1, +1}``."""

    rows: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.rows) > MAX_MATRIX_ORDER:
            raise ValueError("row count exceeds the bounded budget")
        n = len(self.rows[0])
        if n == 0:
            raise ValueError("sign matrix rows must be non-empty")
        for row in self.rows:
            if len(row) != n:
                raise ValueError("sign matrix rows must have equal length")
            if len(row) > MAX_MATRIX_ORDER:
                raise ValueError("column count exceeds the bounded budget")
        _validate_sign_entries(self.rows)
        return self


class HadamardMatrix(StrictModel):
    """A square sign matrix ``H`` satisfying ``H H^T = n I_n`` exactly.

    Orthogonality is a construction invariant of this value.  The validator
    replays the exact Gram product and rejects non-Hadamard matrices.
    """

    rows: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_hadamard(self) -> Self:
        if len(self.rows) > MAX_MATRIX_ORDER:
            raise ValueError("row count exceeds the bounded budget")
        n = len(self.rows)
        for row in self.rows:
            if len(row) != n:
                raise ValueError("Hadamard matrices must be square")
        _validate_sign_entries(self.rows)
        h = [list(row) for row in self.rows]
        for i in range(n):
            for j in range(i, n):
                inner = sum(h[i][k] * h[j][k] for k in range(n))
                expected = n if i == j else 0
                if inner != expected:
                    raise ValueError("Hadamard orthogonality H H^T = n I_n is violated")
        return self


__all__ = [
    "MAX_MATRIX_ORDER",
    "HadamardMatrix",
    "SignMatrix",
]
