"""Typed contracts for the cycle-length profile operation."""

from __future__ import annotations

from pydantic import Field, StrictInt

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
