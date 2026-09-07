"""Typed wire contracts for exact graph transform operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import AfterValidator, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

# Input graph bounds.
MAX_VERTICES = 64
MAX_EDGES = MAX_VERTICES * (MAX_VERTICES - 1) // 2
# A line graph has one vertex per input edge and at most
# (MAX_VERTICES - 2) * MAX_LINE_GRAPH_EDGES = 63,488 edges.
MAX_LINE_GRAPH_EDGES = 1024


def _require_transform_input_graph(
    graph: IndexedSimpleUndirectedGraph,
) -> IndexedSimpleUndirectedGraph:
    if not 0 <= graph.vertex_count <= MAX_VERTICES:
        raise PydanticCustomError(
            "graph.transform_vertex_bound",
            f"graph transforms require between 0 and {MAX_VERTICES} vertices",
        )
    if len(graph.edges) > MAX_EDGES:
        raise PydanticCustomError(
            "graph.transform_edge_bound",
            f"graph transforms support at most {MAX_EDGES} edges",
        )
    return graph


_TransformInputGraph = Annotated[
    IndexedSimpleUndirectedGraph,
    AfterValidator(_require_transform_input_graph),
    Field(description="Canonical graph with 0..64 vertices and at most 2016 edges."),
]


class GraphTransformRequest(StrictModel):
    """One graph transform operation."""

    graph: _TransformInputGraph


class LineGraphRequest(GraphTransformRequest):
    """A line graph whose expanded output fits the canonical graph envelope."""

    graph: _TransformInputGraph = Field(
        description="Canonical graph with 0..64 vertices and at most 1024 edges.",
    )

    @model_validator(mode="after")
    def require_line_graph_output_bound(self) -> Self:
        if len(self.graph.edges) > MAX_LINE_GRAPH_EDGES:
            raise PydanticCustomError(
                "graph.line_graph_edge_bound",
                f"line graphs support at most {MAX_LINE_GRAPH_EDGES} input edges",
            )
        return self


class SubgraphRequest(StrictModel):
    """Extract an induced subgraph on a vertex subset."""

    graph: _TransformInputGraph
    vertices: tuple[int, ...] = Field(min_length=0, max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_valid_vertices(self) -> Self:
        if len(set(self.vertices)) != len(self.vertices):
            raise PydanticCustomError(
                "graph.vertices_must_be_unique", "vertices must be unique"
            )
        for v in self.vertices:
            if not (0 <= v < self.graph.vertex_count):
                raise PydanticCustomError(
                    "graph.vertices_must_be_in_0_vertex_count_1",
                    "vertices must be in 0..vertex_count-1",
                )
        return self
