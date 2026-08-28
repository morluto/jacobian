"""Typed wire contracts for exact graph transform operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import AfterValidator, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

# Input graph bounds.
MAX_VERTICES = 64
MAX_EDGES = 1024

# Result bounds derived from the worst-case transforms over the accepted
# input domain. A line graph reindexes one vertex per input edge, so its
# vertices reach the input edge bound (0..MAX_EDGES-1) rather than the input
# vertex bound; every other transform keeps the input vertex set and stays
# within MAX_VERTICES.
MAX_RESULT_VERTICES = MAX_EDGES
MAX_RESULT_EDGE_ENDPOINT = MAX_EDGES - 1

# |E(L(G))| = sum_v C(deg(v), 2) <= (max_deg - 1) * |E(G)|
#           <= (MAX_VERTICES - 2) * MAX_EDGES.
# Complement, square, and induced subgraph produce at most C(MAX_VERTICES, 2)
# = 2016 edges, so this line-graph bound covers every transform result.
MAX_RESULT_EDGES = MAX_EDGES * (MAX_VERTICES - 2)


class ResultGraphEdge(StrictModel):
    """One undirected edge of a transformed graph.

    Line graph vertices are reindexed input edges, so result endpoints may
    reach the input edge bound, not just the input vertex bound.
    """

    source: int = Field(ge=0, le=MAX_RESULT_EDGE_ENDPOINT)
    target: int = Field(ge=0, le=MAX_RESULT_EDGE_ENDPOINT)

    @model_validator(mode="after")
    def require_distinct(self) -> Self:
        if self.source == self.target:
            raise PydanticCustomError(
                "graph.edge_endpoints_must_be_distinct",
                "edge endpoints must be distinct",
            )
        return self


def _require_transform_input_graph(
    graph: IndexedSimpleUndirectedGraph,
) -> IndexedSimpleUndirectedGraph:
    if not 1 <= graph.vertex_count <= MAX_VERTICES:
        raise PydanticCustomError(
            "graph.transform_vertex_bound",
            f"graph transforms require between 1 and {MAX_VERTICES} vertices",
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
]


class GraphTransformRequest(StrictModel):
    """One graph transform operation."""

    graph: _TransformInputGraph


class GraphResult(StrictModel):
    """The result graph of a transform."""

    vertex_count: int = Field(ge=0, le=MAX_RESULT_VERTICES)
    edges: tuple[ResultGraphEdge, ...] = Field(
        default=(),
        max_length=MAX_RESULT_EDGES,
    )


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
