"""Typed wire contracts for the open-neighbourhood operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, PrivateAttr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.neighborhood._bounds import (
    OpenNeighborhoodAdmission,
    admit_open_neighborhood,
)
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
    _admission: OpenNeighborhoodAdmission = PrivateAttr()

    @model_validator(mode="after")
    def admit_request(self) -> Self:
        try:
            self._admission = admit_open_neighborhood(
                self.graph, self.selected_vertices
            )
        except OperationDomainValidationError as error:
            raise PydanticCustomError(
                "graph.open_neighborhood.request_not_admitted", str(error)
            ) from error
        return self


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
