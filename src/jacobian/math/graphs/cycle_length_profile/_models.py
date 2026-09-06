"""Typed contracts for the cycle-length profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

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
            "profile to fit its retained row and witness bounds."
        )
    )


class CycleLengthRow(StrictModel):
    """One cycle length with a canonical witness cycle."""

    cycle_length: StrictInt = Field(ge=3, le=MAX_VERTICES)
    witness: tuple[str, ...] = Field(min_length=3, max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_canonical_witness(self) -> Self:
        if self.cycle_length != len(self.witness):
            raise PydanticCustomError(
                "cycle_profile.witness_length_mismatch",
                "cycle witness length must match cycle_length",
            )
        if len(set(self.witness)) != len(self.witness):
            raise PydanticCustomError(
                "cycle_profile.witness_vertices_must_be_distinct",
                "cycle witnesses must have distinct vertices",
            )
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
            raise PydanticCustomError(
                "cycle_profile.witness_must_be_canonical",
                "cycle witnesses must use canonical rotation and orientation",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, cycle_length: int, witness: tuple[str, ...]
    ) -> CycleLengthRow:
        """Construct a row after the owner kernel established its invariants."""

        return cls.model_construct(cycle_length=cycle_length, witness=witness)


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CycleLengthRow, ...] = Field(max_length=MAX_VERTICES - 2)

    @model_validator(mode="after")
    def require_structural_profile(self) -> Self:
        lengths = tuple(row.cycle_length for row in self.rows)
        if lengths != tuple(sorted(lengths)) or len(set(lengths)) != len(lengths):
            raise PydanticCustomError(
                "cycle_profile.rows_must_be_sorted_unique",
                "cycle profile rows must be sorted and unique",
            )
        vertices = set(self.graph.vertices)
        if any(not set(row.witness) <= vertices for row in self.rows):
            raise PydanticCustomError(
                "cycle_profile.witness_vertices_must_belong_to_graph",
                "cycle witnesses must use graph vertices",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        rows: tuple[CycleLengthRow, ...],
    ) -> CycleLengthProfileResult:
        """Construct a result after the owner admission and kernel checks."""

        return cls.model_construct(graph=graph, rows=rows)


__all__ = [
    "MAX_VERTICES",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "CycleLengthRow",
]
