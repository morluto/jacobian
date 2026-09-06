"""Typed contracts for the periodic union prefix count operation."""

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, PlainSerializer
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)

# Prefix arithmetic is scalar and does not inherit the 256-digit period bound;
# keep it within the canonical integer representation budget while rejecting
# impractically large conversions before they reach the kernel.
MAX_PREFIX_CUTOFF_DIGITS = CanonicalLimits().max_integer_digits
PeriodicPrefixCutoff = Annotated[
    int,
    BeforeValidator(
        lambda value: _parse_native_integer(value, max_digits=MAX_PREFIX_CUTOFF_DIGITS)
    ),
    PlainSerializer(format_canonical_integer, return_type=str, when_used="json"),
]


def _parse_native_integer(value: Any, *, max_digits: int) -> int:
    if isinstance(value, bool):
        raise PydanticCustomError(
            "canonical_integer.type", "integer must not be boolean"
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = parse_canonical_integer(value)
    else:
        raise PydanticCustomError(
            "canonical_integer.type",
            "integer must be a Python int or canonical decimal string",
        )
    if parsed < 0 or len(format_canonical_integer(parsed).lstrip("-")) > max_digits:
        raise PydanticCustomError(
            "canonical_integer.bounds",
            f"integer must be nonnegative and at most {max_digits} digits",
        )
    return parsed


PeriodicNativeInteger = Annotated[
    int,
    BeforeValidator(
        lambda value: _parse_native_integer(value, max_digits=MAX_PREFIX_CUTOFF_DIGITS)
    ),
    PlainSerializer(format_canonical_integer, return_type=str, when_used="json"),
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
    common_period: PeriodicNativeInteger
    occupied_count: PeriodicNativeInteger
    count: PeriodicNativeInteger


__all__ = [
    "MAX_PREFIX_CUTOFF_DIGITS",
    "PeriodicUnionPrefixCountRequest",
    "PeriodicUnionPrefixCountResult",
]
