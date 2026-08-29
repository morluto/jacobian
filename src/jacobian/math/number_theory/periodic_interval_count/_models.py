"""Typed contracts for the periodic congruence interval count operation."""

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)


class PeriodicIntervalCountRequest(StrictModel):
    """Request to count members of a periodic set in a closed interval."""

    source: PeriodicCongruenceUnionSource
    lower: CanonicalInteger
    upper: CanonicalInteger


class PeriodicIntervalCountResult(StrictModel):
    """The exact count of periodic set members in [lower, upper]."""

    source: PeriodicCongruenceUnionSource
    lower: CanonicalInteger
    upper: CanonicalInteger
    count: CanonicalInteger


__all__ = [
    "PeriodicIntervalCountRequest",
    "PeriodicIntervalCountResult",
]
