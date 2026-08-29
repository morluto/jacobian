"""Provider-independent values for exact combinatorial-matrix operations.

A *sign matrix* is a rectangular matrix whose entries are structurally in
``{-1, +1}``.  A *Hadamard matrix* is a square sign matrix ``H`` satisfying
``H H^T = n I_n`` exactly.

Orthogonality is a construction invariant of the :class:`HadamardMatrix`
value.  Untrusted external JSON first produces a sign matrix; only a
successful exact recognition constructs a :class:`HadamardMatrix`.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_MATERIALIZED_SIGN_MATRIX_AXIS = 1_024
MAX_HADAMARD_ORDER = 512


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by combinatorial-matrix values."""

    return PydanticCustomError(f"combinatorial_matrix.{reason}", message)


def _validate_sign_entries(rows: tuple[tuple[int, ...], ...]) -> None:
    for row in rows:
        for entry in row:
            if entry not in (-1, 1):
                raise _validation_error(
                    "sign_entry_invalid", "sign matrix entries must be -1 or +1"
                )


class SignMatrix(StrictModel):
    """A bounded rectangular matrix whose entries are in ``{-1, +1}``."""

    rows: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.rows) > MAX_MATERIALIZED_SIGN_MATRIX_AXIS:
            raise _validation_error(
                "row_count_exceeds_budget", "row count exceeds the bounded budget"
            )
        n = len(self.rows[0])
        if n == 0:
            raise _validation_error("rows_empty", "sign matrix rows must be non-empty")
        for row in self.rows:
            if len(row) != n:
                raise _validation_error(
                    "row_length_mismatch", "sign matrix rows must have equal length"
                )
            if len(row) > MAX_MATERIALIZED_SIGN_MATRIX_AXIS:
                raise _validation_error(
                    "column_count_exceeds_budget",
                    "column count exceeds the bounded budget",
                )
        _validate_sign_entries(self.rows)
        return self


class HadamardMatrix(StrictModel):
    """A square sign matrix ``H`` satisfying ``H H^T = n I_n`` exactly.

    Deserialization checks only the structural envelope: square shape, sign
    entries, and the admitted order. Exact orthogonality is an owner
    recognition operation. Kernel producers use :meth:`_from_kernel` after
    that invariant is established.
    """

    rows: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_structural_square_sign_matrix(self) -> Self:
        if len(self.rows) > MAX_HADAMARD_ORDER:
            raise _validation_error(
                "row_count_exceeds_budget", "row count exceeds the bounded budget"
            )
        n = len(self.rows)
        for row in self.rows:
            if len(row) != n:
                raise _validation_error(
                    "not_square", "Hadamard matrices must be square"
                )
        _validate_sign_entries(self.rows)
        return self

    @classmethod
    def _from_kernel(cls, *, rows: tuple[tuple[int, ...], ...]) -> Self:
        """Construct after exact orthogonality or a Hadamard-preserving map."""

        return cls.model_construct(rows=rows)


__all__ = [
    "MAX_HADAMARD_ORDER",
    "MAX_MATERIALIZED_SIGN_MATRIX_AXIS",
    "HadamardMatrix",
    "SignMatrix",
]
