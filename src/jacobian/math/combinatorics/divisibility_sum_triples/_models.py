"""Typed contracts for the divisibility-sum triples hypergraph operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_DIVISIBILITY_SUM_INTERVAL_SIZE = 42


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triples hypergraph."""

    lower: int = Field(ge=1)
    upper: int

    @model_validator(mode="after")
    def require_bounded_interval(self) -> Self:
        if self.upper < self.lower:
            raise PydanticCustomError(
                "divisibility_sum_triples.interval_order",
                "upper must be at least lower",
            )
        if self.upper - self.lower + 1 > MAX_DIVISIBILITY_SUM_INTERVAL_SIZE:
            raise PydanticCustomError(
                "divisibility_sum_triples.interval_size",
                "divisibility-sum triples admit at most 42 interval values",
            )
        return self


class DivisibilitySumTriplesResult(StrictModel):
    """The canonical divisibility-sum triples hypergraph."""

    lower: int
    upper: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_DIVISIBILITY_SUM_INTERVAL_SIZE",
    "DivisibilitySumTriplesRequest",
    "DivisibilitySumTriplesResult",
]
