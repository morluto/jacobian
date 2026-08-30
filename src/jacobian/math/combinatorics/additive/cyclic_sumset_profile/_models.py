"""Typed contracts for the cyclic sumset representation profile operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json

MAX_CYCLIC_SUMSET_PAIRS = 100_000


def _result_wire_bytes(
    modulus: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int:
    pair_count = len(left) * len(right)
    support_upper = min(modulus, pair_count)
    entries = [
        {"residue": str(modulus - 1), "count": str(pair_count)}
        for _ in range(support_upper)
    ]
    return len(
        encode_strict_json(
            {
                "modulus": str(modulus),
                "left": [str(value) for value in left],
                "right": [str(value) for value in right],
                "entries": entries,
                "support_cardinality": str(support_upper),
            },
            limits=CanonicalLimits(max_output_bytes=1 << 60),
        )
    )


class CyclicSumsetRequest(StrictModel):
    """Request the cyclic sumset representation profile."""

    modulus: int = Field(gt=0)
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
        if (
            _result_wire_bytes(self.modulus, self.left, self.right)
            > CanonicalLimits().max_output_bytes
        ):
            raise PydanticCustomError(
                "cyclic_sumset.result_bytes_exceeded",
                "cyclic sumset profile exceeds the canonical output-byte limit",
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
    "MAX_CYCLIC_SUMSET_PAIRS",
    "CyclicSumsetEntry",
    "CyclicSumsetRequest",
    "CyclicSumsetResult",
    "_result_wire_bytes",
]
