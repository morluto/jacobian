"""Typed contracts for the rational fixed-arity sum profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_RESULT_ROWS = 1_000_000


class RationalFixedAritySumRequest(StrictModel):
    """Request for the rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...]
    arity: StrictInt = Field(ge=0)


class SumProfileRow(StrictModel):
    """One attained rational sum with its multiplicity."""

    sum_value: CanonicalRational
    multiplicity: StrictInt = Field(ge=1)


class RationalFixedAritySumResult(StrictModel):
    """The complete rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...]
    arity: StrictInt = Field(ge=0)
    rows: tuple[SumProfileRow, ...] = Field(max_length=MAX_RESULT_ROWS)

    @classmethod
    def _from_kernel(
        cls,
        values: tuple[CanonicalRational, ...],
        arity: int,
        rows: tuple[SumProfileRow, ...],
    ) -> Self:
        """Construct a result after the owner has established its profile."""
        return cls.model_construct(values=values, arity=arity, rows=rows)


__all__ = [
    "MAX_RESULT_ROWS",
    "RationalFixedAritySumRequest",
    "RationalFixedAritySumResult",
    "SumProfileRow",
]
