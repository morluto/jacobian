"""Typed contracts owned by elementary counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics._models import _combinatorics_validation_error

MAX_BINOMIAL_N = 10_000
_MAX_PARTS = 256


class BinomialRequest(StrictModel):
    """A wider safe bound for Python's efficient exact ``math.comb`` path."""

    n: StrictInt = Field(ge=0, le=MAX_BINOMIAL_N)
    k: StrictInt = Field(ge=0, le=MAX_BINOMIAL_N)


class IntegerListRequest(StrictModel):
    """A bounded list of nonnegative parts for multinomial counting."""

    values: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=_MAX_PARTS)

    @model_validator(mode="after")
    def require_nonnegative_parts(self) -> Self:
        if any(parse_canonical_integer(value) < 0 for value in self.values):
            raise _combinatorics_validation_error(
                "integer list values must be nonnegative"
            )
        return self
