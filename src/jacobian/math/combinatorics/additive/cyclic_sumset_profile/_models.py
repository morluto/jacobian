"""Typed contracts for the cyclic sumset representation profile operation."""

from jacobian._models import StrictModel


class CyclicSumsetRequest(StrictModel):
    """Request the cyclic sumset representation profile."""

    modulus: int
    left: tuple[int, ...]
    right: tuple[int, ...]


class CyclicSumsetEntry(StrictModel):
    """One residue and its representation count."""

    residue: int
    count: int


class CyclicSumsetResult(StrictModel):
    """The complete cyclic sumset representation profile."""

    modulus: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    entries: tuple[CyclicSumsetEntry, ...]
    support_cardinality: int


__all__ = [
    "CyclicSumsetEntry",
    "CyclicSumsetRequest",
    "CyclicSumsetResult",
]
