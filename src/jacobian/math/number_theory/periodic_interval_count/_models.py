"""Typed contracts for the periodic congruence interval count operation."""

from typing import Annotated

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)

MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS = 100_000
PeriodicIntervalEndpoint = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS)
]
PeriodicIntervalCount = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS + 1)
]


class PeriodicIntervalCountRequest(StrictModel):
    """Request to count members of a periodic set in a closed interval."""

    source: PeriodicCongruenceUnionSource
    lower: PeriodicIntervalEndpoint
    upper: PeriodicIntervalEndpoint


class PeriodicIntervalCountResult(StrictModel):
    """The exact count of periodic set members in [lower, upper]."""

    source: PeriodicCongruenceUnionSource
    lower: PeriodicIntervalEndpoint
    upper: PeriodicIntervalEndpoint
    count: PeriodicIntervalCount


__all__ = [
    "PeriodicIntervalCountRequest",
    "PeriodicIntervalCountResult",
]
