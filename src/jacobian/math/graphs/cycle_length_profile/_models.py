"""Typed contracts for the cycle-length profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

MAX_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES


class CycleLengthProfileRequest(StrictModel):
    """Request for the simple-cycle length profile of a graph."""

    graph: SimpleUndirectedGraph


class CycleLengthRow(StrictModel):
    """One cycle length with a canonical witness cycle."""

    cycle_length: StrictInt = Field(ge=3, le=MAX_VERTICES)
    witness: tuple[str, ...] = Field(min_length=3, max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_matching_witness_length(self) -> Self:
        if self.cycle_length != len(self.witness):
            raise ValueError("cycle witness length must match cycle_length")
        if len(set(self.witness)) != len(self.witness):
            raise ValueError("cycle witnesses must have distinct vertices")
        return self


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CycleLengthRow, ...] = Field(max_length=MAX_VERTICES - 2)

    @model_validator(mode="after")
    def require_sorted_unique_lengths(self) -> Self:
        if len(self.graph.vertices) > MAX_VERTICES:
            raise ValueError(f"cycle profiles support at most {MAX_VERTICES} vertices")
        lengths = tuple(row.cycle_length for row in self.rows)
        if lengths != tuple(sorted(lengths)) or len(set(lengths)) != len(lengths):
            raise ValueError("cycle profile rows must be sorted and unique")
        vertices = set(self.graph.vertices)
        edges = {frozenset(edge) for edge in self.graph.edges}
        for row in self.rows:
            if not set(row.witness) <= vertices:
                raise ValueError("cycle witnesses must use graph vertices")
            witness_edges = {
                frozenset(
                    (row.witness[index], row.witness[(index + 1) % row.cycle_length])
                )
                for index in range(row.cycle_length)
            }
            if not witness_edges <= edges:
                raise ValueError("cycle witnesses must use graph edges")
        return self


__all__ = [
    "MAX_VERTICES",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "CycleLengthRow",
]
