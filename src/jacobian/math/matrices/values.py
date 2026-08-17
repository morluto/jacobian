"""Provider-independent exact matrix values."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

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


class RationalMatrix(StrictModel):
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


class IntegerMatrix(StrictModel):
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


class SmithNormalForm(StrictModel):
    """A backend-independent positive divisibility diagonal and its metadata."""

    normal_form: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    invariant_factors: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_MATRIX_DIMENSION
    )
    transformation_available: Literal[False] = False
    convention: Literal["POSITIVE_DIVISIBILITY_DIAGONAL"] = (
        "POSITIVE_DIVISIBILITY_DIAGONAL"
    )

    @model_validator(mode="after")
    def require_invariant_factor_chain(self) -> Self:
        rows = len(self.normal_form.entries)
        columns = len(self.normal_form.entries[0])
        if len(self.invariant_factors) != self.rank:
            raise ValueError("nonzero invariant factor count must equal rank")
        if self.rank > min(rows, columns):
            raise ValueError("Smith rank cannot exceed the matrix dimensions")
        factors = tuple(
            parse_canonical_integer(value) for value in self.invariant_factors
        )
        if any(value <= 0 for value in factors):
            raise ValueError("Smith invariant factors must be positive")
        if any(right % left != 0 for left, right in pairwise(factors)):
            raise ValueError("each Smith invariant factor must divide the next")
        for row, entries in enumerate(self.normal_form.entries):
            for column, value in enumerate(entries):
                expected = factors[row] if row == column and row < self.rank else 0
                if parse_canonical_integer(value) != expected:
                    raise ValueError(
                        "Smith normal form must contain its positive invariant "
                        "factors on the leading diagonal and zero elsewhere"
                    )
        return self

    @field_validator("invariant_factors")
    @classmethod
    def require_bounded_invariant_factors(
        cls, values: tuple[CanonicalInteger, ...]
    ) -> tuple[CanonicalInteger, ...]:
        for value in values:
            if len(value.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS:
                raise ValueError(
                    f"matrix scalars are limited to {MAX_MATRIX_SCALAR_DIGITS} decimal digits"
                )
        return values


__all__ = [
    "MAX_MATRIX_DIMENSION",
    "MAX_MATRIX_SCALAR_DIGITS",
    "IntegerMatrix",
    "RationalMatrix",
    "SmithNormalForm",
    "require_matrix_scalar_digits",
]
