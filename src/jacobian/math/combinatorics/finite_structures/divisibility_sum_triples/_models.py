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


def _validate_interval_shape(lower_bound: int, upper_bound: int) -> None:
    if not isinstance(lower_bound, int) or isinstance(lower_bound, bool):
        raise PydanticCustomError(
            "divisibility_sum.bound_type", "interval bounds must be integers"
        )
    if not isinstance(upper_bound, int) or isinstance(upper_bound, bool):
        raise PydanticCustomError(
            "divisibility_sum.bound_type", "interval bounds must be integers"
        )
    if lower_bound > upper_bound:
        raise PydanticCustomError(
            "divisibility_sum.invalid_bounds",
            "lower_bound must not exceed upper_bound",
        )


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triple hypergraph."""

    lower_bound: int
    upper_bound: int

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        _validate_interval_shape(self.lower_bound, self.upper_bound)
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
    "_validate_interval_shape",
]
