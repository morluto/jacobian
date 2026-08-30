"""Typed contracts for the periodic union prefix count operation."""

from math import lcm
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, parse_canonical_integer
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
    cutoff: PeriodicPrefixCutoff
    common_period: PeriodicPositiveInteger
    occupied_count: PeriodicNonnegativeInteger
    count: PeriodicNonnegativeInteger

    @model_validator(mode="after")
    def require_count_invariants(self) -> Self:
        cutoff = parse_canonical_integer(self.cutoff)
        period = parse_canonical_integer(self.common_period)
        occupied = parse_canonical_integer(self.occupied_count)
        count = parse_canonical_integer(self.count)
        source_period = lcm(
            *(parse_canonical_integer(subset.modulus) for subset in self.source.subsets)
        )
        if period != source_period:
            raise PydanticCustomError(
                "number_theory.periodic_prefix.period_mismatch",
                "common_period must equal the source common period",
            )
        if occupied > period:
            raise PydanticCustomError(
                "number_theory.periodic_prefix.occupied_count_out_of_range",
                "occupied_count must lie between 0 and common_period",
            )
        if count > cutoff:
            raise PydanticCustomError(
                "number_theory.periodic_prefix.count_out_of_range",
                "count must lie between 0 and cutoff",
            )
        remainder = cutoff % period
        partial = count - (cutoff // period) * occupied
        if not 0 <= partial <= remainder:
            raise PydanticCustomError(
                "number_theory.periodic_prefix.count_identity_mismatch",
                "count must decompose into full periods and a bounded partial period",
            )
        return self


__all__ = [
    "MAX_PREFIX_CUTOFF_DIGITS",
    "PeriodicUnionPrefixCountRequest",
    "PeriodicUnionPrefixCountResult",
]
