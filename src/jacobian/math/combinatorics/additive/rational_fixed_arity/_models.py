"""Typed contracts for the rational fixed-arity sum profile operation."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_SEQUENCE_LENGTH = 1_000
MAX_ARITY = MAX_SEQUENCE_LENGTH
MAX_RESULT_ROWS = 1_000_000


class RationalFixedAritySumRequest(StrictModel):
    """Request for the rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...] = Field(max_length=MAX_SEQUENCE_LENGTH)
    arity: StrictInt = Field(ge=0, le=MAX_ARITY)


class SumProfileRow(StrictModel):
    """One attained rational sum with its multiplicity."""

    sum_value: CanonicalRational
    multiplicity: StrictInt = Field(ge=1)


class RationalFixedAritySumResult(StrictModel):
    """The complete rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...] = Field(max_length=MAX_SEQUENCE_LENGTH)
    arity: StrictInt = Field(ge=0, le=MAX_ARITY)
    rows: tuple[SumProfileRow, ...] = Field(max_length=MAX_RESULT_ROWS)

    @model_validator(mode="after")
    def require_profile_invariants(self) -> Self:
        sums = tuple(row.sum_value.as_fraction() for row in self.rows)
        if sums != tuple(sorted(sums)) or len(set(sums)) != len(sums):
            raise ValueError("sum profile rows must be sorted and unique")
        expected = (
            comb(len(self.values), self.arity) if self.arity <= len(self.values) else 0
        )
        if sum(row.multiplicity for row in self.rows) != expected:
            raise ValueError("sum profile multiplicities must cover every index tuple")
        return self


__all__ = [
    "MAX_ARITY",
    "MAX_RESULT_ROWS",
    "MAX_SEQUENCE_LENGTH",
    "RationalFixedAritySumRequest",
    "RationalFixedAritySumResult",
    "SumProfileRow",
]
