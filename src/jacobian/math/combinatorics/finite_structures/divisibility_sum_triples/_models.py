"""Typed contracts for divisibility-sum triple hypergraph construction."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_INTERVAL_SIZE = 200


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triple hypergraph."""

    lower_bound: int
    upper_bound: int

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower_bound > self.upper_bound:
            raise PydanticCustomError(
                "divisibility_sum.invalid_bounds",
                "lower_bound must not exceed upper_bound",
            )
        interval_size = self.upper_bound - self.lower_bound + 1
        if interval_size > MAX_INTERVAL_SIZE:
            raise PydanticCustomError(
                "divisibility_sum.interval_too_large",
                f"interval size must not exceed {MAX_INTERVAL_SIZE}",
            )
        return self


class DivisibilitySumTriplesResult(StrictModel):
    """The divisibility-sum triple hypergraph."""

    lower_bound: int
    upper_bound: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_INTERVAL_SIZE",
    "DivisibilitySumTriplesRequest",
    "DivisibilitySumTriplesResult",
]
