"""Typed wire contracts for directed graph operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


class DirectedGraph(StrictModel):
    """A simple directed graph for reachability, SCC, and related analyses.

    ``vertex_count`` carries a conservative admission fallback bounding
    linear-time traversal work and vertex-sized results; consumers with
    sharper derived budgets refine admission further.
    """

    vertex_count: int = Field(ge=2, le=256)
    edges: tuple[tuple[int, int], ...] = Field(max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if source == target:
                raise PydanticCustomError(
                    "graph.self_loops_are_not_allowed", "self-loops are not allowed"
                )
            endpoint_pair = (source, target)
            if endpoint_pair in seen:
                raise PydanticCustomError(
                    "graph.directed_edges_must_be_unique",
                    "directed edges must be unique",
                )
            seen.add(endpoint_pair)
        return self


class ReachabilityRequest(StrictModel):
    graph: DirectedGraph
    source: int = Field(ge=0, le=255)

    @model_validator(mode="after")
    def require_valid_source(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.source_must_be_in_0_graph_vertex_count_1",
                "source must be in 0..graph.vertex_count-1",
            )
        return self


class ReachabilityResult(StrictModel):
    source: int = Field(ge=0, le=255)
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
                raise PydanticCustomError(
                    "graph.acyclic_order_must_list_every_vertex",
                    "acyclic order must list every vertex",
                )
        elif self.order:
            raise PydanticCustomError(
                "graph.cyclic_graph_must_not_report_a_topological_order",
                "cyclic graph must not report a topological order",
            )
        return self
