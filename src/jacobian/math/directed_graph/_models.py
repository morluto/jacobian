"""Typed wire contracts for directed graph operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class DirectedGraph(StrictModel):
    """A simple directed graph for reachability, SCC, and related analyses."""

    vertex_count: int = Field(ge=2, le=64)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=512)

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
            endpoint_pair = (source, target)
            if endpoint_pair in seen:
                raise ValueError("directed edges must be unique")
            seen.add(endpoint_pair)
        return self


class ReachabilityRequest(StrictModel):
    graph: DirectedGraph
    source: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_source(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise ValueError("source must be in 0..graph.vertex_count-1")
        return self


class ReachabilityResult(StrictModel):
    source: int = Field(ge=0, le=63)
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    convention: Literal["NETWORKX_DESCENDANTS"] = "NETWORKX_DESCENDANTS"


class StronglyConnectedComponentsRequest(StrictModel):
    graph: DirectedGraph


class StronglyConnectedComponentsResult(StrictModel):
    component_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]
    convention: Literal["NETWORKX_STRONGLY_CONNECTED_COMPONENTS"] = (
        "NETWORKX_STRONGLY_CONNECTED_COMPONENTS"
    )


class CondensationRequest(StrictModel):
    graph: DirectedGraph


class CondensationEdge(StrictModel):
    source: int = Field(ge=0)
    target: int = Field(ge=0)


class CondensationResult(StrictModel):
    vertex_count: int = Field(ge=0, strict=True)
    components: tuple[tuple[int, ...], ...]
    edges: tuple[CondensationEdge, ...] = Field(default=())
    convention: Literal["NETWORKX_CONDENSATION"] = "NETWORKX_CONDENSATION"


class AcyclicOrderRequest(StrictModel):
    graph: DirectedGraph


class AcyclicOrderResult(StrictModel):
    acyclic: bool
    order: tuple[int, ...]
    convention: Literal["NETWORKX_TOPOLOGICAL_SORT"] = "NETWORKX_TOPOLOGICAL_SORT"

    @model_validator(mode="after")
    def require_order_matches_acyclicity(self) -> Self:
        if self.acyclic:
            if not self.order:
                raise ValueError("acyclic order must list every vertex")
        elif self.order:
            raise ValueError("cyclic graph must not report a topological order")
        return self
