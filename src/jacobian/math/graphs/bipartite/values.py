"""Canonical indexed bipartite graphs with ordered, fixed sides."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

Vertex = Annotated[int, Field(ge=0, le=1023)]


class FixedBipartiteGraph(StrictModel):
    """An indexed graph with explicit ordered left and right vertex sets.

    The sides partition 0..vertex_count-1. Every edge crosses sides; isolated
    vertices and empty sides retain their chosen interpretation. Side order
    is mathematical context and is never inferred by two-colouring.
    """

    graph: IndexedSimpleUndirectedGraph
    left_vertices: tuple[Vertex, ...] = Field(max_length=1024)
    right_vertices: tuple[Vertex, ...] = Field(max_length=1024)

    @model_validator(mode="after")
    def require_fixed_partition(self) -> Self:
        vertices = (*self.left_vertices, *self.right_vertices)
        if len(vertices) != self.graph.vertex_count or set(vertices) != set(
            range(self.graph.vertex_count)
        ):
            raise ValueError("left and right must partition all graph vertices")
        left = set(self.left_vertices)
        if any((a in left) == (b in left) for a, b in self.graph.edges):
            raise ValueError("every graph edge must cross the declared sides")
        return self


class BipartiteVertexRegion(StrictModel):
    """Vertex subsets on fixed left/right sides; ambient graph is retained by the owner."""

    left_vertices: tuple[Vertex, ...] = Field(max_length=1024)
    right_vertices: tuple[Vertex, ...] = Field(max_length=1024)
