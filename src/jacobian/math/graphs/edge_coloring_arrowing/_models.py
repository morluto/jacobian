"""Typed contracts for edge-colouring Ramsey arrowing decision."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_HOST_VERTICES = 10
MAX_HOST_EDGES = 45
MAX_TARGET_VERTICES = 8


class EdgeColoringArrowingRequest(StrictModel):
    """Decide whether a host graph arrows a tuple of target graphs."""

    host_graph: SimpleUndirectedGraph
    targets: tuple[SimpleUndirectedGraph, ...]

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.host_graph.vertices) > MAX_HOST_VERTICES:
            raise PydanticCustomError(
                "graph.arrowing.host_too_large",
                f"host graph has at most {MAX_HOST_VERTICES} vertices",
            )
        if len(self.host_graph.edges) > MAX_HOST_EDGES:
            raise PydanticCustomError(
                "graph.arrowing.host_too_many_edges",
                f"host graph has at most {MAX_HOST_EDGES} edges",
            )
        for i, target in enumerate(self.targets):
            if len(target.vertices) > MAX_TARGET_VERTICES:
                raise PydanticCustomError(
                    "graph.arrowing.target_too_large",
                    f"target {i} has at most {MAX_TARGET_VERTICES} vertices",
                )
            if not target.vertices:
                raise PydanticCustomError(
                    "graph.arrowing.target_empty",
                    f"target {i} must not be empty",
                )
        return self


class EdgeColoringArrowingResult(StrictModel):
    """Result of an edge-colouring Ramsey arrowing decision."""

    host_graph: SimpleUndirectedGraph
    targets: tuple[SimpleUndirectedGraph, ...]
    outcome: str  # "ARROWS" or "DOES_NOT_ARROW"
    avoiding_coloring: tuple[tuple[int, int], ...] | None = (
        None  # (edge_index, color) pairs
    )


__all__ = [
    "MAX_HOST_EDGES",
    "MAX_HOST_VERTICES",
    "MAX_TARGET_VERTICES",
    "EdgeColoringArrowingRequest",
    "EdgeColoringArrowingResult",
]
