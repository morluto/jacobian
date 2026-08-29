"""Typed contracts for the cycle-length profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 16


class CycleLengthProfileRequest(StrictModel):
    """Request for the simple-cycle length profile of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "cycle_profile.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        return self


class CycleLengthRow(StrictModel):
    """One cycle length with a canonical witness cycle."""

    cycle_length: int
    witness: tuple[str, ...]


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CycleLengthRow, ...]


__all__ = [
    "MAX_VERTICES",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "CycleLengthRow",
]
