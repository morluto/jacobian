"""Typed contracts for the weighted monotone endpoint profile operation."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class WeightedMonotoneProfileRequest(StrictModel):
    """Request the weighted monotone endpoint profiles."""

    alphabet: tuple[int, ...]
    weights: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_weight_axis(self) -> Self:
        if len(self.weights) != len(self.alphabet):
            raise PydanticCustomError(
                "weighted_monotone.weight_axis",
                "weights must align one-for-one with the alphabet axis",
            )
        return self


class WeightedMonotoneProfileResult(StrictModel):
    """The two exact endpoint DP profiles."""

    alphabet: tuple[int, ...]
    weights: tuple[CanonicalRational, ...]
    increasing_profile: tuple[CanonicalRational, ...]
    decreasing_profile: tuple[CanonicalRational, ...]


__all__ = [
    "WeightedMonotoneProfileRequest",
    "WeightedMonotoneProfileResult",
]
