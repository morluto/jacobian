"""Typed contracts for the rational subset-sum profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_SEQUENCE_LENGTH = 20


class RationalSubsetSumRequest(StrictModel):
    """Request for the rational subset-sum profile."""

    values: tuple[CanonicalRational, ...]


class SubsetSumRow(StrictModel):
    """One attained rational sum with its multiplicity."""

    sum_value: CanonicalRational
    multiplicity: int


class RationalSubsetSumResult(StrictModel):
    """The complete rational subset-sum profile."""

    values: tuple[CanonicalRational, ...]
    rows: tuple[SubsetSumRow, ...]


__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "RationalSubsetSumRequest",
    "RationalSubsetSumResult",
    "SubsetSumRow",
]
