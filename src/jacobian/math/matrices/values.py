"""Provider-independent values for exact integer matrix normal forms."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian.canonical import parse_canonical_integer
from jacobian.contracts.exact import CanonicalInteger
from jacobian.contracts.matrices import (
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
)
from jacobian.contracts.results import ContractModel


class SmithNormalForm(ContractModel):
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


__all__ = ["SmithNormalForm"]
