"""Typed wire contracts for graph isomorphism decision operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class UndirectedGraph(ContractModel):
    """A simple undirected graph given by a vertex count and edge list.

    Vertices are the integers ``0..vertex_count-1``.  Edges are unordered
    pairs with no self-loops or duplicates.
    """

    vertex_count: int = Field(ge=1, le=32)
    edges: tuple[tuple[int, int], ...] = Field(max_length=512)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("a simple graph cannot contain self-loops")
            key = (min(u, v), max(u, v))
            if key in seen:
                raise ValueError("a simple graph cannot contain duplicate edges")
            seen.add(key)
        return self


class GraphIsomorphismRequest(ContractModel):
    """Two simple undirected graphs to compare for isomorphism."""

    graph_a: UndirectedGraph
    graph_b: UndirectedGraph

    @model_validator(mode="after")
    def require_matching_vertex_counts(self) -> Self:
        if self.graph_a.vertex_count != self.graph_b.vertex_count:
            raise ValueError(
                "graph_a and graph_b must have the same vertex count"
            )
        return self


class GraphIsomorphismResult(ContractModel):
    """Decision and explicit certificate for a graph isomorphism query.

    When ``decision`` is ``ISOMORPHIC`` the optional ``mapping`` is a vertex
    bijection from the first graph to the second graph.  When ``decision`` is
    ``NOT_ISOMORPHIC`` the ``mapping`` is ``None``.
    """

    decision: Literal["ISOMORPHIC", "NOT_ISOMORPHIC"]
    mapping: tuple[tuple[int, int], ...] | None = None
    convention: Literal["NETWORKX_IS_ISOMORPHIC"] = "NETWORKX_IS_ISOMORPHIC"

    @model_validator(mode="after")
    def require_mapping_when_isomorphic(self) -> Self:
        if self.decision == "ISOMORPHIC" and self.mapping is None:
            raise ValueError("an isomorphic decision must carry a mapping")
        if self.decision == "NOT_ISOMORPHIC" and self.mapping is not None:
            raise ValueError("a non-isomorphic decision must not carry a mapping")
        return self
