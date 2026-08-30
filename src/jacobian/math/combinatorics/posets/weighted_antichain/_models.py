"""Typed contracts for the maximum weight antichain operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    FinitePoset,
)

MAX_ENUMERATION_WORK = 250_000_000


class MaximumWeightAntichainRequest(StrictModel):
    """Request for the maximum weight antichain of a poset."""

    poset: FinitePoset
    weights: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_POSET_ELEMENTS,
        description=(
            "One nonnegative exact rational per poset element. Exhaustive "
            "admission uses the candidate-subset work envelope."
        ),
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.weights) != len(self.poset.elements):
            raise PydanticCustomError(
                "weighted_antichain.weight_count_mismatch",
                "weights must have exactly one entry per poset element",
            )
        for w in self.weights:
            if w.num.startswith("-") and w.num != "0":
                raise PydanticCustomError(
                    "weighted_antichain.negative_weight",
                    "all weights must be nonnegative",
                )
        return self


class MaximumWeightAntichainResult(StrictModel):
    """The maximum weight antichain of a poset."""

    poset: FinitePoset
    weights: tuple[CanonicalRational, ...]
    maximum_weight: CanonicalRational
    antichain: tuple[str, ...]


__all__ = [
    "MAX_ENUMERATION_WORK",
    "MaximumWeightAntichainRequest",
    "MaximumWeightAntichainResult",
]
