"""Typed wire contracts for exact graph transform operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

MAX_VERTICES = 64
MAX_EDGES = 1024


class GraphEdge(ContractModel):
    """One undirected edge."""

    source: int = Field(ge=0, le=MAX_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_VERTICES - 1)

    @model_validator(mode="after")
    def require_distinct(self) -> Self:
        if self.source == self.target:
            raise ValueError("edge endpoints must be distinct")
        return self


class SimpleGraph(ContractModel):
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
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            key = (
                (edge.source, edge.target)
                if edge.source < edge.target
                else (edge.target, edge.source)
            )
            if key in seen:
                raise ValueError("edges must be unique")
            seen.add(key)
        return self


class GraphTransformRequest(ContractModel):
    """One graph transform operation."""

    graph: SimpleGraph


class GraphResult(ContractModel):
    """The result graph of a transform."""

    vertex_count: int = Field(ge=1, le=MAX_VERTICES * MAX_VERTICES)
    edges: tuple[GraphEdge, ...] = Field(default=(), max_length=MAX_EDGES)
    method: Literal["NETWORKX"] = "NETWORKX"


class SubgraphRequest(ContractModel):
    """Extract an induced subgraph on a vertex subset."""

    graph: SimpleGraph
    vertices: tuple[int, ...] = Field(min_length=0, max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_valid_vertices(self) -> Self:
        for v in self.vertices:
            if not (0 <= v < self.graph.vertex_count):
                raise ValueError("vertices must be in 0..vertex_count-1")
        return self


__all__ = [
    "GraphEdge",
    "GraphResult",
    "GraphTransformRequest",
    "SimpleGraph",
    "SubgraphRequest",
]
