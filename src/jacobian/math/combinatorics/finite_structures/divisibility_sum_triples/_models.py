"""Typed contracts for divisibility-sum triple hypergraph construction."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)

MAX_INTERVAL_SIZE = 200


def _validate_interval(lower_bound: int, upper_bound: int) -> None:
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
    interval_size = upper_bound - lower_bound + 1
    if interval_size > MAX_INTERVAL_SIZE:
        raise PydanticCustomError(
            "divisibility_sum.interval_too_large",
            f"interval size must not exceed {MAX_INTERVAL_SIZE}",
        )
    triple_count = comb(interval_size, 3) if interval_size >= 3 else 0
    if triple_count > MAX_EDGES or 3 * triple_count > MAX_TOTAL_INCIDENCES:
        raise PydanticCustomError(
            "divisibility_sum.output_too_large",
            "the potential triple family exceeds the hypergraph envelope",
        )


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triple hypergraph."""

    lower_bound: int
    upper_bound: int

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        _validate_interval(self.lower_bound, self.upper_bound)
        return self


class DivisibilitySumTriplesResult(StrictModel):
    """The divisibility-sum triple hypergraph."""

    lower_bound: int
    upper_bound: int
    hypergraph: FiniteHypergraph

    @model_validator(mode="after")
    def require_source_and_edge_contract(self) -> Self:
        _validate_interval(self.lower_bound, self.upper_bound)
        expected_vertices = tuple(
            str(value) for value in range(self.lower_bound, self.upper_bound + 1)
        )
        if self.hypergraph.vertices != expected_vertices:
            raise PydanticCustomError(
                "divisibility_sum.vertex_source_mismatch",
                "hypergraph vertices must equal the requested interval",
            )
        for edge_id, members in self.hypergraph.edges:
            if len(members) != 3:
                raise PydanticCustomError(
                    "divisibility_sum.edge_not_triple",
                    f"edge {edge_id} must contain exactly three vertices",
                )
            values = tuple(sorted(int(member) for member in members))
            if (
                not values[0] < values[1] < values[2]
                or (values[1] + values[2]) % values[0] != 0
            ):
                raise PydanticCustomError(
                    "divisibility_sum.invalid_edge",
                    f"edge {edge_id} is not a divisibility-sum triple",
                )
        return self


__all__ = [
    "MAX_INTERVAL_SIZE",
    "DivisibilitySumTriplesRequest",
    "DivisibilitySumTriplesResult",
    "_validate_interval",
]
