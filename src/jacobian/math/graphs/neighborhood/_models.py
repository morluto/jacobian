"""Typed wire contracts for the open-neighbourhood operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

MAX_NEIGHBORHOOD_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
MAX_NEIGHBORHOOD_EDGES = MAX_INDEXED_SIMPLE_GRAPH_EDGES


class NeighborhoodRequest(StrictModel):
    """Compute the exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...] = Field(
        min_length=0,
        max_length=MAX_NEIGHBORHOOD_VERTICES,
    )

    @model_validator(mode="after")
    def validate_selected_vertices(self) -> Self:
        vertex_set = set(self.graph.vertices)
        for v in self.selected_vertices:
            if v not in vertex_set:
                raise PydanticCustomError(
                    "graph.selected_vertex_not_in_graph",
                    "every selected vertex must be a declared graph vertex",
                )
        if len(set(self.selected_vertices)) != len(self.selected_vertices):
            raise PydanticCustomError(
                "graph.selected_vertices_must_be_unique",
                "selected vertices must be unique",
            )
        return self


class NeighborhoodResult(StrictModel):
    """The exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        vertex_set = set(self.graph.vertices)
        for v in self.neighborhood:
            if v not in vertex_set:
                raise PydanticCustomError(
                    "graph.neighborhood_vertex_not_in_graph",
                    "every neighbourhood vertex must be a declared graph vertex",
                )
        return self


__all__ = [
    "MAX_NEIGHBORHOOD_EDGES",
    "MAX_NEIGHBORHOOD_VERTICES",
    "NeighborhoodRequest",
    "NeighborhoodResult",
]
