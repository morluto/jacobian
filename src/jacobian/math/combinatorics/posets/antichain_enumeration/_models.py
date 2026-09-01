"""Typed contracts for the antichain enumeration operation."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import FinitePoset

MAX_ELEMENTS = 24
MAX_ANTICHAINS = 50_000


def require_antichain_enumeration_envelope(
    poset: FinitePoset,
    min_cardinality: int,
    max_cardinality: int,
) -> int:
    n = len(poset.elements)
    if n > MAX_ELEMENTS:
        raise ValueError(
            f"antichain enumeration supports at most {MAX_ELEMENTS} elements"
        )
    if min_cardinality < 0 or max_cardinality < min_cardinality:
        raise ValueError(
            "antichain cardinalities must form a nonnegative ordered range"
        )
    upper = min(max_cardinality, n)
    candidates = sum(comb(n, size) for size in range(min_cardinality, upper + 1))
    if candidates > MAX_ANTICHAINS:
        raise ValueError(
            f"antichain enumeration exceeds the {MAX_ANTICHAINS}-candidate bound"
        )
    return candidates


class AntichainEnumerationRequest(StrictModel):
    """Request to enumerate antichains of specified cardinalities."""

    poset: FinitePoset
    min_cardinality: int = Field(default=1, ge=0, le=MAX_ELEMENTS)
    max_cardinality: int = Field(default=1, ge=0, le=MAX_ELEMENTS)

    @model_validator(mode="after")
    def require_ordered_cardinality_range(self) -> Self:
        if self.max_cardinality < self.min_cardinality:
            raise PydanticCustomError(
                "poset.antichain_cardinality_range",
                "max_cardinality must be at least min_cardinality",
            )
        return self


class AntichainEnumerationResult(StrictModel):
    """A complete enumeration of antichains in the requested cardinality range."""

    poset_digest: Sha256Digest
    min_cardinality: int = Field(ge=0, le=MAX_ELEMENTS)
    max_cardinality: int = Field(ge=0, le=MAX_ELEMENTS)
    antichains: tuple[tuple[str, ...], ...] = Field(max_length=MAX_ANTICHAINS)
    count: int = Field(ge=0, le=MAX_ANTICHAINS)


__all__ = [
    "MAX_ANTICHAINS",
    "MAX_ELEMENTS",
    "AntichainEnumerationRequest",
    "AntichainEnumerationResult",
    "require_antichain_enumeration_envelope",
]
