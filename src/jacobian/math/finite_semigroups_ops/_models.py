"""Typed wire contracts for finite semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ELEMENTS = 64


def _validate_table(multiplication_table: tuple[tuple[int, ...], ...]) -> int:
    n = len(multiplication_table)
    if n == 0 or n > MAX_ELEMENTS:
        raise ValueError("multiplication table must have between 1 and 64 rows")
    for row in multiplication_table:
        if len(row) != n:
            raise ValueError("multiplication table must be square")
        if any(not 0 <= v < n for v in row):
            raise ValueError("table entries must be in 0..n-1")
    return n


class FiniteSemigroup(StrictModel):
    """A finite semigroup given by its multiplication table."""

    multiplication_table: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_table(self.multiplication_table)
        return self


class ElementPowerRequest(StrictModel):
    multiplication_table: tuple[tuple[int, ...], ...] = Field(min_length=1)
    element: int = Field(ge=0)
    exponent: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = _validate_table(self.multiplication_table)
        if self.element >= n:
            raise ValueError("element must be in 0..n-1")
        return self


class IdempotentsRequest(StrictModel):
    multiplication_table: tuple[tuple[int, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_table(self.multiplication_table)
        return self


class GeneratedSubsemigroupRequest(StrictModel):
    multiplication_table: tuple[tuple[int, ...], ...] = Field(min_length=1)
    generators: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = _validate_table(self.multiplication_table)
        if any(not 0 <= g < n for g in self.generators):
            raise ValueError("generators must be in 0..n-1")
        return self


class PrincipalIdealsRequest(StrictModel):
    multiplication_table: tuple[tuple[int, ...], ...] = Field(min_length=1)
    elements: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = _validate_table(self.multiplication_table)
        if any(not 0 <= e < n for e in self.elements):
            raise ValueError("elements must be in 0..n-1")
        return self


class ElementPowerResult(StrictModel):
    result: int = Field(ge=0)
    method: str = "ITERATED_MULTIPLICATION"


class IdempotentsResult(StrictModel):
    idempotents: tuple[int, ...]
    method: str = "FIXED_POINT_CHECK"


class GeneratedSubsemigroupResult(StrictModel):
    elements: tuple[int, ...]
    size: int = Field(ge=1)
    method: str = "CLOSURE_ENUMERATION"


class PrincipalIdealsResult(StrictModel):
    ideals: tuple[tuple[int, ...], ...]
    method: str = "SET_MULTIPLICATION"
