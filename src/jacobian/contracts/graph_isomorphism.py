"""Typed wire contracts for graph isomorphism decision operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class SimpleGraph(ContractModel):
    """A simple graph (no parallel edges, no self-loops) for isomorphism.

    ``directed`` selects whether the graph is treated as directed or
    undirected.  For undirected graphs the validator canonicalises each
    edge so ``(u, v)`` and ``(v, u)`` describe the same adjacency.
    """

    vertex_count: int = Field(ge=1, le=64)
    directed: bool = False
    edges: tuple[tuple[int, int], ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if source == target:
                raise ValueError("self-loops are not allowed")
            if self.directed:
                edge_key = (source, target)
            else:
                edge_key = (min(source, target), max(source, target))
            if edge_key in seen:
                raise ValueError("edges must be unique")
            seen.add(edge_key)
        return self


class GraphIsomorphismRequest(ContractModel):
    """Request an isomorphism decision between two simple graphs.

    The two graphs must agree on directedness and on their vertex count.
    """

    graph_a: SimpleGraph
    graph_b: SimpleGraph

    @model_validator(mode="after")
    def require_consistent_directedness(self) -> Self:
        if self.graph_a.directed != self.graph_b.directed:
            raise ValueError("both graphs must have the same directedness")
        return self

    @model_validator(mode="after")
    def require_consistent_vertex_count(self) -> Self:
        if self.graph_a.vertex_count != self.graph_b.vertex_count:
            raise ValueError("both graphs must have the same vertex count")
        return self


class VertexMappingPair(ContractModel):
    """One ``(from_vertex, to_vertex)`` entry in an isomorphism witness."""

    from_vertex: int = Field(ge=0, le=63)
    to_vertex: int = Field(ge=0, le=63)


class GraphIsomorphismResult(ContractModel):
    """The result of a graph isomorphism decision.

    When ``status`` is ``ISOMORPHIC`` the ``vertex_mapping`` field carries an
    explicit bijection as a list of ``(from_vertex, to_vertex)`` pairs that
    the caller can independently verify.  When ``status`` is
    ``NOT_ISOMORPHIC`` the ``vertex_mapping`` field is empty.
    """

    status: Literal["ISOMORPHIC", "NOT_ISOMORPHIC"]
    vertex_mapping: tuple[VertexMappingPair, ...] = Field(default=())
    convention: Literal["NETWORKX_IS_ISOMORPHIC"] = "NETWORKX_IS_ISOMORPHIC"
