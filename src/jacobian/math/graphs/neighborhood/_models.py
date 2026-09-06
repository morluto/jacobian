"""Typed wire contracts for the open-neighbourhood operation."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
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
        return canonicalize_json_containers(value)


class NeighborhoodResult(StrictModel):
    """The exact open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]

    @model_validator(mode="after")
    def validate_result(self) -> NeighborhoodResult:
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

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        selected_vertices: tuple[str, ...],
        neighborhood: tuple[str, ...],
    ) -> NeighborhoodResult:
        """Construct a result after the kernel established its relation."""

        return cls.model_construct(
            graph=graph,
            selected_vertices=selected_vertices,
            neighborhood=neighborhood,
        )


__all__ = [
    "NeighborhoodRequest",
    "NeighborhoodResult",
]
