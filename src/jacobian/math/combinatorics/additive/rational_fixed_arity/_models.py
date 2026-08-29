"""Typed contracts for the rational fixed-arity sum profile operation."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_SEQUENCE_LENGTH = 200
MAX_ARITY = 10


class RationalFixedAritySumRequest(StrictModel):
    """Request for the rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...]
    arity: int


class SumProfileRow(StrictModel):
    """One attained rational sum with its multiplicity."""

    sum_value: CanonicalRational
    multiplicity: int


class RationalFixedAritySumResult(StrictModel):
    """The complete rational fixed-arity sum profile."""

    values: tuple[CanonicalRational, ...]
    arity: int
    rows: tuple[SumProfileRow, ...]


__all__ = [
    "MAX_ARITY",
    "MAX_SEQUENCE_LENGTH",
    "RationalFixedAritySumRequest",
    "RationalFixedAritySumResult",
    "SumProfileRow",
]
