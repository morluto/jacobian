"""Typed wire contracts for finite-dimensional algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DIM = 32
MAX_ENTRIES = 1024


class StructureConstants(StrictModel):
    """Structure constants c[i][j][k] for a finite-dimensional algebra."""

    dimension: int = Field(ge=1, le=MAX_DIM)
    field_order: int = Field(ge=2, le=251)
    multiplication: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ENTRIES
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if len(self.multiplication) != self.dimension:
            raise ValueError("multiplication must have dimension rows")
        for row in self.multiplication:
            if len(row) != self.dimension:
                raise ValueError("multiplication must be square")
            if any(not 0 <= v < self.field_order for v in row):
                raise ValueError("entries must be canonical field residues")
        return self


# Requests


class CenterRequest(StrictModel):
    algebra: StructureConstants


class RadicalRequest(StrictModel):
    algebra: StructureConstants


# Results


class CenterResult(StrictModel):
    center_basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    center_dimension: int = Field(ge=1)
    method: str = "COMMUTANT_COMPUTATION"


class RadicalResult(StrictModel):
    radical_basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    is_semisimple: bool
    method: str = "JACOBSON_RADICAL"
