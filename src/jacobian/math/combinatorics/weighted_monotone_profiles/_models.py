"""Typed contracts for the weighted monotone endpoint profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class WeightedMonotoneProfileRequest(StrictModel):
    """Request the weighted monotone endpoint profiles."""

    alphabet: tuple[int, ...]
    weights: tuple[CanonicalRational, ...]


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
