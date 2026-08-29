"""Typed contracts for the rational subset-sum profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class RationalSubsetSumRequest(StrictModel):
    """Request the indexed rational subset-sum profile."""

    values: tuple[CanonicalRational, ...]


class RationalSubsetSumEntry(StrictModel):
    """One attainable sum and its multiplicity."""

    sum: CanonicalRational
    multiplicity: int


class RationalSubsetSumResult(StrictModel):
    """The complete indexed rational subset-sum profile."""

    values: tuple[CanonicalRational, ...]
    entries: tuple[RationalSubsetSumEntry, ...]
    support_cardinality: int


__all__ = [
    "RationalSubsetSumEntry",
    "RationalSubsetSumRequest",
    "RationalSubsetSumResult",
]
