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

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical simple graph. Admission also requires the complete first-"
            "witness search to fit the 10,000,000-unit work bound and the complete "
            "profile to fit the canonical output envelope."
        )
    )


class CycleLengthRow(StrictModel):
    """One cycle length with a canonical witness cycle."""

    cycle_length: StrictInt = Field(ge=3, le=MAX_VERTICES)
    witness: tuple[str, ...] = Field(min_length=3, max_length=MAX_VERTICES)

    @classmethod
    def _from_kernel(cls, cycle_length: int, witness: tuple[str, ...]) -> Self:
        """Construct a row after the owner kernel established its invariants."""

        return cls.model_construct(cycle_length=cycle_length, witness=witness)

    @model_validator(mode="after")
    def require_matching_witness_length(self) -> Self:
        if self.cycle_length != len(self.witness):
            raise ValueError("cycle witness length must match cycle_length")
        if len(set(self.witness)) != len(self.witness):
            raise ValueError("cycle witnesses must have distinct vertices")
        rotations = [
            self.witness[index:] + self.witness[:index]
            for index in range(len(self.witness))
        ]
        reversed_witness = (self.witness[0], *reversed(self.witness[1:]))
        rotations.extend(
            reversed_witness[index:] + reversed_witness[:index]
            for index in range(len(self.witness))
        )
        if self.witness != min(rotations):
            raise ValueError(
                "cycle witnesses must use canonical rotation and orientation"
            )
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

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        rows: tuple[CycleLengthRow, ...],
    ) -> Self:
        """Construct a result after the owner admission and kernel checks."""

        return cls.model_construct(graph=graph, rows=rows)


__all__ = [
    "MAX_VERTICES",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "CycleLengthRow",
]
