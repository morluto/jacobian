"""Typed wire contracts for exact graph transform operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

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


class GraphEdge(StrictModel):
    """One undirected edge of an input graph."""

    source: int = Field(ge=0, le=MAX_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_VERTICES - 1)

    @model_validator(mode="after")
    def require_distinct(self) -> Self:
        if self.source == self.target:
            raise PydanticCustomError(
                "graph.edge_endpoints_must_be_distinct",
                "edge endpoints must be distinct",
            )
        return self


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


class SimpleGraph(StrictModel):
    """A finite simple undirected graph."""

    vertex_count: int = Field(ge=1, le=MAX_VERTICES)
    edges: tuple[GraphEdge, ...] = Field(default=(), max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            if not (
                0 <= edge.source < self.vertex_count
                and 0 <= edge.target < self.vertex_count
            ):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            key = (
                (edge.source, edge.target)
                if edge.source < edge.target
                else (edge.target, edge.source)
            )
            if key in seen:
                raise PydanticCustomError(
                    "graph.edges_must_be_unique", "edges must be unique"
                )
            seen.add(key)
        return self


class GraphTransformRequest(StrictModel):
    """One graph transform operation."""

    graph: SimpleGraph


class GraphResult(StrictModel):
    """The result graph of a transform."""

    vertex_count: int = Field(ge=0, le=MAX_RESULT_VERTICES)
    edges: tuple[ResultGraphEdge, ...] = Field(
        default=(),
        max_length=MAX_RESULT_EDGES,
    )
    method: Literal["NETWORKX"] = "NETWORKX"


class SubgraphRequest(StrictModel):
    """Extract an induced subgraph on a vertex subset."""

    graph: SimpleGraph
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
