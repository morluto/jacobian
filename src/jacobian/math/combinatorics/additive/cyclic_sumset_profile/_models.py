"""Typed contracts for the cyclic sumset representation profile operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_CYCLIC_SUMSET_PAIRS = 100_000
MAX_CYCLIC_SUMSET_MODULUS = (1 << 53) - 1


class CyclicSumsetRequest(StrictModel):
    """Request the cyclic sumset representation profile."""

    modulus: int = Field(gt=0, le=MAX_CYCLIC_SUMSET_MODULUS)
    left: tuple[int, ...]
    right: tuple[int, ...]

    @model_validator(mode="after")
    def require_canonical_bounded_operands(self) -> Self:
        if len(self.left) * len(self.right) > MAX_CYCLIC_SUMSET_PAIRS:
            raise PydanticCustomError(
                "cyclic_sumset.pair_work_exceeded",
                "cyclic sumset exceeds the 100000-pair work bound",
            )
        if any(not 0 <= value < self.modulus for value in (*self.left, *self.right)):
            raise PydanticCustomError(
                "cyclic_sumset.canonical_residue",
                "cyclic sumset operands must be canonical residues modulo modulus",
            )
        if len(set(self.left)) != len(self.left) or len(set(self.right)) != len(
            self.right
        ):
            raise PydanticCustomError(
                "cyclic_sumset.duplicate_operand",
                "cyclic sumset operands must contain distinct residues",
            )
        return self


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
    "MAX_CYCLIC_SUMSET_MODULUS",
    "MAX_CYCLIC_SUMSET_PAIRS",
    "CyclicSumsetEntry",
    "CyclicSumsetRequest",
    "CyclicSumsetResult",
]
