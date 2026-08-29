"""Typed contracts owned by elementary counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics._models import _combinatorics_validation_error
from jacobian.math.combinatorics.operations import (
    MAX_MULTINOMIAL_PARTS,
    MAX_MULTINOMIAL_TOTAL,
    MAX_SPARSE_COUNTING_INDEX,
)


class SparseCountingPairRequest(StrictModel):
    """Indices for result-sensitive exact binomial-product counting."""

    n: StrictInt = Field(ge=0, le=MAX_SPARSE_COUNTING_INDEX)
    k: StrictInt = Field(ge=0, le=MAX_SPARSE_COUNTING_INDEX)


class IntegerListRequest(StrictModel):
    """A bounded list of nonnegative parts for multinomial counting."""

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_MULTINOMIAL_PARTS
    )

    @model_validator(mode="after")
    def require_nonnegative_parts(self) -> Self:
        parts = tuple(parse_canonical_integer(value) for value in self.values)
        if any(value < 0 for value in parts):
            raise _combinatorics_validation_error(
                "integer list values must be nonnegative"
            )
        if len(parts) > 1 and sum(parts) > MAX_MULTINOMIAL_TOTAL:
            raise _combinatorics_validation_error(
                "the sum of integer list values exceeds the "
                f"{MAX_MULTINOMIAL_TOTAL}-element counting bound"
            )
        return self
