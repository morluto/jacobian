"""Typed contracts for the periodic union prefix count operation."""

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)


class PeriodicUnionPrefixCountRequest(StrictModel):
    """Request for the prefix count of a periodic congruence union."""

    source: PeriodicCongruenceUnionSource
    cutoff: CanonicalInteger


class PeriodicUnionPrefixCountResult(StrictModel):
    """The exact count of integers in [1, cutoff] belonging to the periodic set."""

    source: PeriodicCongruenceUnionSource
    cutoff: CanonicalInteger
    common_period: CanonicalInteger
    occupied_count: CanonicalInteger
    count: CanonicalInteger


__all__ = [
    "PeriodicUnionPrefixCountRequest",
    "PeriodicUnionPrefixCountResult",
]
