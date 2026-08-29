"""Typed contracts for the cyclic prefix-sum residue profile operation."""

from __future__ import annotations

from jacobian._models import StrictModel

MAX_SEQUENCE_LENGTH = 10_000
MAX_MODULUS_DIGITS = 100


class CyclicPrefixSumResidueProfileRequest(StrictModel):
    """Request for the cyclic prefix-sum residue profile."""

    sequence: tuple[int, ...]
    modulus: int


class PrefixSumResidueRow(StrictModel):
    """One row of the residue profile."""

    residue: int
    positions: tuple[int, ...]


class CyclicPrefixSumResidueProfileResult(StrictModel):
    """The complete cyclic prefix-sum residue profile."""

    modulus: int
    rows: tuple[PrefixSumResidueRow, ...]


__all__ = [
    "MAX_MODULUS_DIGITS",
    "MAX_SEQUENCE_LENGTH",
    "CyclicPrefixSumResidueProfileRequest",
    "CyclicPrefixSumResidueProfileResult",
    "PrefixSumResidueRow",
]
