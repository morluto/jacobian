"""Typed contracts for the periodic union prefix count operation."""

from typing import Annotated

from pydantic import Field, StringConstraints

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory._periodic_models import (
    MAX_PERIODIC_INTEGER_DIGITS,
    PeriodicCongruenceUnionSource,
)

MAX_PREFIX_CUTOFF_DIGITS = MAX_PERIODIC_INTEGER_DIGITS
PeriodicPrefixCutoff = Annotated[
    str,
    StringConstraints(
        pattern=rf"^(?:0|[1-9][0-9]{{0,{MAX_PREFIX_CUTOFF_DIGITS - 1}}})$",
        max_length=MAX_PREFIX_CUTOFF_DIGITS,
        strict=True,
    ),
]


class PeriodicUnionPrefixCountRequest(StrictModel):
    """Request for the prefix count of a periodic congruence union."""

    source: PeriodicCongruenceUnionSource
    cutoff: PeriodicPrefixCutoff = Field(
        description=(
            "Nonnegative canonical decimal cutoff with at most "
            f"{MAX_PREFIX_CUTOFF_DIGITS} digits."
        ),
        examples=["6"],
    )


class PeriodicUnionPrefixCountResult(StrictModel):
    """The exact count of integers in [1, cutoff] belonging to the periodic set."""

    source: PeriodicCongruenceUnionSource
    cutoff: CanonicalInteger
    common_period: CanonicalInteger
    occupied_count: CanonicalInteger
    count: CanonicalInteger


__all__ = [
    "MAX_PREFIX_CUTOFF_DIGITS",
    "PeriodicUnionPrefixCountRequest",
    "PeriodicUnionPrefixCountResult",
]
