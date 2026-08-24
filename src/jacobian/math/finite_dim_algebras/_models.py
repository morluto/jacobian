"""Typed wire contracts for finite-dimensional algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DIM = 128
MAX_ENTRIES = (
    MAX_DIM * MAX_DIM
)  # 16384; outer tensor bound (n <= 128, logical n^3 entries)


class StructureConstants(StrictModel):
    """Structure constants ``c[i][j][k]`` for a finite-dimensional algebra.

    The algebra has basis ``e_0, ..., e_{n-1}`` over the prime field ``F_q`` and

    ``e_i * e_j = sum_k c[i][j][k] e_k``

    where every ``c[i][j][k]`` is a canonical residue in ``{0, ..., q - 1}``.
    """

    dimension: int = Field(ge=1, le=MAX_DIM)
    field_order: int = Field(ge=2, le=251)
    multiplication: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_ENTRIES
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        n = self.dimension
        if len(self.multiplication) != n:
            raise ValueError("multiplication must have dimension rows")
        for row in self.multiplication:
            if len(row) != n:
                raise ValueError(
                    "multiplication must be square in the first two indices"
                )
            for inner in row:
                if len(inner) != n:
                    raise ValueError(
                        "multiplication must be a 3-index tensor c[i][j][k]"
                    )
                if any(not 0 <= v < self.field_order for v in inner):
                    raise ValueError("entries must be canonical field residues")
        return self


# Requests


class CenterRequest(StrictModel):
    algebra: StructureConstants


# Results


class CenterResult(StrictModel):
    center_basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    center_dimension: int = Field(ge=0)
    method: str = "COMMUTANT_COMPUTATION"
