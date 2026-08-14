"""Exact rational matrix operation contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)

MAX_MATRIX_DIMENSION = 32
MAX_MATRIX_SCALAR_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


def require_matrix_scalar_digits(
    entries: tuple[tuple[str | CanonicalRational, ...], ...],
    *,
    maximum: int,
    label: str,
) -> None:
    """Apply an operation-owned scalar budget to an authoritative matrix value."""

    for row in entries:
        for value in row:
            components = (value,) if isinstance(value, str) else (value.num, value.den)
            if any(len(component.lstrip("-")) > maximum for component in components):
                raise ValueError(
                    f"{label} scalars are limited to {maximum} decimal digits"
                )


class RationalMatrix(ContractModel):
    """One nonempty rectangular matrix over canonical rationals."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    entries: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise ValueError("matrix rows must contain between 1 and 32 entries")
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self


class IntegerMatrix(ContractModel):
    """One nonempty rectangular matrix over exact canonical integers."""

    matrix_schema_version: Literal["1"] = "1"
    domain: Literal["ZZ"] = "ZZ"
    entries: tuple[tuple[CanonicalInteger, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_MATRIX_DIMENSION:
            raise ValueError("matrix rows must contain between 1 and 32 entries")
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        require_matrix_scalar_digits(
            self.entries,
            maximum=MAX_MATRIX_SCALAR_DIGITS,
            label="matrix",
        )
        return self
