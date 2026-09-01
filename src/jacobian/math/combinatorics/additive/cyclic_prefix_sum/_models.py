"""Typed contracts for the cyclic prefix-sum residue profile operation."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    indexed_sequence_item_ceiling,
)

MAX_SEQUENCE_LENGTH = 10_000
MAX_MODULUS_DIGITS = 100

BoundedModulus = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_MODULUS_DIGITS, strict=True),
]


class CyclicPrefixSumResidueProfileRequest(StrictModel):
    """Request for the cyclic prefix-sum residue profile."""

    sequence: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_SEQUENCE_LENGTH)),
    ]
    modulus: BoundedModulus


class PrefixSumResidueRow(StrictModel):
    """One row of the residue profile."""

    residue: CanonicalInteger
    positions: tuple[StrictInt, ...] = Field(min_length=1)


class CyclicPrefixSumResidueProfileResult(StrictModel):
    """The complete cyclic prefix-sum residue profile."""

    modulus: BoundedModulus
    rows: tuple[PrefixSumResidueRow, ...] = Field(max_length=MAX_SEQUENCE_LENGTH)

    @model_validator(mode="after")
    def require_sorted_unique_rows(self) -> Self:
        residues = tuple(int(row.residue) for row in self.rows)
        if residues != tuple(sorted(residues)) or len(set(residues)) != len(residues):
            raise ValueError("residue rows must be sorted and unique")
        positions = [position for row in self.rows for position in row.positions]
        if len(positions) > MAX_SEQUENCE_LENGTH:
            raise ValueError("residue rows contain too many prefix positions")
        if any(
            position < 1 or position > MAX_SEQUENCE_LENGTH for position in positions
        ):
            raise ValueError("prefix positions must be within the sequence bound")
        return self


__all__ = [
    "MAX_MODULUS_DIGITS",
    "MAX_SEQUENCE_LENGTH",
    "BoundedModulus",
    "CyclicPrefixSumResidueProfileRequest",
    "CyclicPrefixSumResidueProfileResult",
    "PrefixSumResidueRow",
]
