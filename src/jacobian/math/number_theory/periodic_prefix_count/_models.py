"""Typed contracts for the periodic union prefix count operation."""

from typing import Annotated

from pydantic import Field

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
    PeriodicNonnegativeInteger,
    PeriodicPositiveInteger,
)

# Prefix arithmetic is scalar and does not inherit the 256-digit period bound;
# keep it within the canonical integer representation budget while rejecting
# impractically large conversions before they reach the kernel.
MAX_PREFIX_CUTOFF_DIGITS = CanonicalLimits().max_integer_digits
PeriodicPrefixCutoff = Annotated[
    ExactInteger,
    Field(
        ge=0,
        json_schema_extra={
            "pattern": rf"^(?:0|[1-9][0-9]{{0,{MAX_PREFIX_CUTOFF_DIGITS - 1}}})(?![\s\S])",
            "maxLength": MAX_PREFIX_CUTOFF_DIGITS,
        },
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
    cutoff: PeriodicPrefixCutoff
    common_period: PeriodicPositiveInteger
    occupied_count: PeriodicNonnegativeInteger
    count: PeriodicPrefixCutoff


__all__ = [
    "MAX_PREFIX_CUTOFF_DIGITS",
    "PeriodicUnionPrefixCountRequest",
    "PeriodicUnionPrefixCountResult",
]
