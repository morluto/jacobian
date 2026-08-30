"""Typed contracts for the periodic union prefix count operation."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)


class PeriodicUnionPrefixCountRequest(StrictModel):
    """Request for the prefix count of a periodic congruence union."""

    source: PeriodicCongruenceUnionSource
    cutoff: CanonicalInteger

    @model_validator(mode="after")
    def require_nonnegative_cutoff(self) -> Self:
        if parse_canonical_integer(self.cutoff) < 0:
            raise PydanticCustomError(
                "periodic_prefix_count.nonnegative_cutoff",
                "periodic prefix cutoff must be nonnegative",
            )
        return self


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
