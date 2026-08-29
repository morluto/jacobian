"""Typed wire contracts for the open-neighbourhood operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)


class NeighborhoodRequest(StrictModel):
    """Compute the exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES
    )
class NeighborhoodResult(StrictModel):
    """The exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        selected = set(self.selected_vertices)
        neighborhood = set(self.neighborhood)
        if self.selected_vertices != tuple(
            vertex for vertex in self.graph.vertices if vertex in selected
        ):
            raise PydanticCustomError(
                "graph.selected_vertices_must_be_canonical",
                "selected vertices must be a subset in source-vertex order",
            )
        if self.neighborhood != tuple(
            vertex for vertex in self.graph.vertices if vertex in neighborhood
        ):
            raise PydanticCustomError(
                "graph.neighborhood_must_be_canonical",
                "neighbourhood vertices must be a subset in source-vertex order",
            )
        if selected & neighborhood:
            raise PydanticCustomError(
                "graph.open_neighborhood_must_exclude_selected_vertices",
                "an open neighbourhood must exclude selected vertices",
            )
        return self


__all__ = [
    "NeighborhoodRequest",
    "NeighborhoodResult",
]
