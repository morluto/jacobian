"""Typed contracts for the weighted-antichain operation."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import FinitePoset

MAX_WEIGHTED_ANTICHAIN_ELEMENTS = 16


class WeightedAntichainRequest(StrictModel):
    """Request the maximum-weight antichain of a finite poset."""

    poset: FinitePoset
    weights: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_WEIGHTED_ANTICHAIN_ELEMENTS
    )

    @model_validator(mode="after")
    def require_aligned_bounded_weights(self) -> Self:
        if len(self.poset.elements) > MAX_WEIGHTED_ANTICHAIN_ELEMENTS:
            raise PydanticCustomError(
                "poset.weighted_antichain_work_exceeded",
                "weighted antichain search supports at most 16 elements",
            )
        if len(self.weights) != len(self.poset.elements):
            raise PydanticCustomError(
                "poset.weighted_antichain_weight_axis",
                "weights must align one-for-one with the poset element axis",
            )
        return self


class WeightedAntichainResult(StrictModel):
    """The exact maximum-weight antichain."""

    poset_digest: str
    weights: tuple[CanonicalRational, ...]
    maximum_weight: CanonicalRational
    maximum_antichain: tuple[str, ...]
    method: Literal["EXACT_BOUNDED_SUBSET_SEARCH"]


__all__ = [
    "MAX_WEIGHTED_ANTICHAIN_ELEMENTS",
    "WeightedAntichainRequest",
    "WeightedAntichainResult",
]
