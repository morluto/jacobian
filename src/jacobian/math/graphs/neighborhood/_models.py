"""Typed wire contracts for the open-neighbourhood operation."""

from __future__ import annotations

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

    @model_validator(mode="before")
    @classmethod
    def bound_raw_selected_vertices(cls, value: object) -> object:
        if isinstance(value, dict):
            selected = value.get("selected_vertices")
            if (
                isinstance(selected, (list, tuple))
                and len(selected) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES
            ):
                raise PydanticCustomError(
                    "graph.selected_vertices_bound",
                    "selected vertices exceed the raw tuple-length bound",
                )
        return value


class NeighborhoodResult(StrictModel):
    """The exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]


__all__ = [
    "NeighborhoodRequest",
    "NeighborhoodResult",
]
