"""Typed wire contracts for graph flow and cut operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class CapacitatedEdge(ContractModel):
    """One directed edge with a rational capacity."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    capacity: CanonicalRational


class FlowGraph(ContractModel):
    """A directed capacitated graph for flow problems."""

    vertex_count: int = Field(ge=2, le=64)
    edges: tuple[CapacitatedEdge, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_valid_vertices(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            if not (
                0 <= edge.source < self.vertex_count
                and 0 <= edge.target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if edge.capacity.as_fraction() < 0:
                raise ValueError("edge capacities must be nonnegative")
            endpoint_pair = (edge.source, edge.target)
            if endpoint_pair in seen:
                raise ValueError("directed edges must be unique")
            seen.add(endpoint_pair)
        return self


class MaxFlowRequest(ContractModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise ValueError("source must be in 0..graph.vertex_count-1")
        if not (0 <= self.sink < self.graph.vertex_count):
            raise ValueError("sink must be in 0..graph.vertex_count-1")
        if self.source == self.sink:
            raise ValueError("source and sink must be distinct")
        return self


class FlowEdgeValue(ContractModel):
    """The flow assigned to one directed edge."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    flow: CanonicalRational


class MaxFlowResult(ContractModel):
    flow_value: CanonicalRational
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    flow_edges: tuple[FlowEdgeValue, ...] = Field(default=())
    convention: Literal["NETWORKX_MAXIMUM_FLOW"] = "NETWORKX_MAXIMUM_FLOW"


class MinCutRequest(ContractModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise ValueError("source must be in 0..graph.vertex_count-1")
        if not (0 <= self.sink < self.graph.vertex_count):
            raise ValueError("sink must be in 0..graph.vertex_count-1")
        if self.source == self.sink:
            raise ValueError("source and sink must be distinct")
        return self


class MinCutResult(ContractModel):
    cut_value: CanonicalRational
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    convention: Literal["NETWORKX_MINIMUM_CUT"] = "NETWORKX_MINIMUM_CUT"


class EdgeDisjointPathsGraph(ContractModel):
    """A simple directed graph for edge-disjoint path computation."""

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


class EdgeDisjointPathsRequest(ContractModel):
    graph: EdgeDisjointPathsGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise ValueError("source must be in 0..graph.vertex_count-1")
        if not (0 <= self.sink < self.graph.vertex_count):
            raise ValueError("sink must be in 0..graph.vertex_count-1")
        if self.source == self.sink:
            raise ValueError("source and sink must be distinct")
        return self


class EdgeDisjointPathsResult(ContractModel):
    path_count: int = Field(ge=0)
    paths: tuple[tuple[int, ...], ...] = Field(default=())
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    convention: Literal["NETWORKX_EDGE_DISJOINT_PATHS"] = "NETWORKX_EDGE_DISJOINT_PATHS"
