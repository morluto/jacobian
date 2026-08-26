"""Shared Pydantic wire contracts for exact combinatorics operations."""

from __future__ import annotations

from pydantic import Field, StrictInt
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel


def _combinatorics_validation_error(message: str) -> PydanticCustomError:
    """Return the stable structured reason used by owner-local validators."""

    lowered = message.lower()
    code = "combinatorics.invariant"
    for marker, candidate in (
        ("partition", "combinatorics.partition_invariant"),
        ("requested index", "combinatorics.result_bound"),
        ("rational", "combinatorics.rational_bound"),
        ("result", "combinatorics.result_bound"),
    ):
        if marker in lowered:
            code = candidate
            break
    return PydanticCustomError(code, message, {})


_MAX_N = 10_000


class NonnegativeIntegerRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)


class NonnegativePairRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=_MAX_N)
    k: StrictInt = Field(ge=0, le=_MAX_N)


class IntegerResult(StrictModel):
    value: CanonicalInteger


class RationalResult(StrictModel):
    value: CanonicalRational
