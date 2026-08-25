"""Typed wire contracts for graph flow and cut operations."""

from __future__ import annotations

from math import lcm
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

# Derived integer scales that make rational capacities and costs exact
# integers are intermediate growth, not input size: each denominator is
# already bounded by the canonical rational limit, but the least common
# multiple of up to 1,024 such denominators can grow far beyond what the
# integer backend can expand. Requests whose derived scale exceeds this
# documented conservative digit budget are rejected before any backend graph
# is constructed.
MAX_MIN_COST_FLOW_DERIVED_SCALE_DIGITS = 4096


def _bounded_denominator_scale(denominators: tuple[int, ...], kind: str) -> int:
    """Return the LCM of ``denominators`` under the derived-scale digit budget."""
    scale = 1
    for denominator in denominators:
        scale = lcm(scale, abs(denominator))
        if len(str(scale)) > MAX_MIN_COST_FLOW_DERIVED_SCALE_DIGITS:
            raise PydanticCustomError(
                "graph.least_common_multiple_kind_denominators_exceeds_max",
                f"the least common multiple of {kind} denominators exceeds the "
                f"{MAX_MIN_COST_FLOW_DERIVED_SCALE_DIGITS}-digit derived-scale limit",
            )
    return scale


class CapacitatedEdge(StrictModel):
    """One directed edge with a rational capacity."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    capacity: CanonicalRational


class FlowGraph(StrictModel):
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
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if edge.capacity.as_fraction() < 0:
                raise PydanticCustomError(
                    "graph.edge_capacities_must_be_nonnegative",
                    "edge capacities must be nonnegative",
                )
            endpoint_pair = (edge.source, edge.target)
            if endpoint_pair in seen:
                raise PydanticCustomError(
                    "graph.directed_edges_must_be_unique",
                    "directed edges must be unique",
                )
            seen.add(endpoint_pair)
        return self


class MaxFlowRequest(StrictModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.source_must_be_in_0_graph_vertex_count_1",
                "source must be in 0..graph.vertex_count-1",
            )
        if not (0 <= self.sink < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.sink_must_be_in_0_graph_vertex_count_1",
                "sink must be in 0..graph.vertex_count-1",
            )
        if self.source == self.sink:
            raise PydanticCustomError(
                "graph.source_and_sink_must_be_distinct",
                "source and sink must be distinct",
            )
        return self


class FlowEdgeValue(StrictModel):
    """The flow assigned to one directed edge."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    flow: CanonicalRational


class MaxFlowResult(StrictModel):
    flow_value: CanonicalRational
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    flow_edges: tuple[FlowEdgeValue, ...] = Field(default=())
    convention: Literal["NETWORKX_MAXIMUM_FLOW"] = "NETWORKX_MAXIMUM_FLOW"


class MinCutRequest(StrictModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.source_must_be_in_0_graph_vertex_count_1",
                "source must be in 0..graph.vertex_count-1",
            )
        if not (0 <= self.sink < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.sink_must_be_in_0_graph_vertex_count_1",
                "sink must be in 0..graph.vertex_count-1",
            )
        if self.source == self.sink:
            raise PydanticCustomError(
                "graph.source_and_sink_must_be_distinct",
                "source and sink must be distinct",
            )
        return self


class MinCutResult(StrictModel):
    cut_value: CanonicalRational
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    convention: Literal["NETWORKX_MINIMUM_CUT"] = "NETWORKX_MINIMUM_CUT"


class EdgeDisjointPathsGraph(StrictModel):
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


class EdgeDisjointPathsRequest(StrictModel):
    graph: EdgeDisjointPathsGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_valid_terminals(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.source_must_be_in_0_graph_vertex_count_1",
                "source must be in 0..graph.vertex_count-1",
            )
        if not (0 <= self.sink < self.graph.vertex_count):
            raise PydanticCustomError(
                "graph.sink_must_be_in_0_graph_vertex_count_1",
                "sink must be in 0..graph.vertex_count-1",
            )
        if self.source == self.sink:
            raise PydanticCustomError(
                "graph.source_and_sink_must_be_distinct",
                "source and sink must be distinct",
            )
        return self


class EdgeDisjointPathsResult(StrictModel):
    path_count: int = Field(ge=0)
    paths: tuple[tuple[int, ...], ...] = Field(default=())
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    convention: Literal["NETWORKX_EDGE_DISJOINT_PATHS"] = "NETWORKX_EDGE_DISJOINT_PATHS"


class CostedFlowEdge(StrictModel):
    """One directed edge with a capacity and a cost per unit of flow."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    capacity: CanonicalRational
    cost: CanonicalRational


class CostedFlowGraph(StrictModel):
    """A directed graph with capacities and per-unit costs for flow problems."""

    vertex_count: int = Field(ge=2, le=64)
    edges: tuple[CostedFlowEdge, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
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
            if edge.capacity.as_fraction() < 0:
                raise PydanticCustomError(
                    "graph.edge_capacities_must_be_nonnegative",
                    "edge capacities must be nonnegative",
                )
            endpoint_pair = (edge.source, edge.target)
            if endpoint_pair in seen:
                raise PydanticCustomError(
                    "graph.directed_edges_must_be_unique",
                    "directed edges must be unique",
                )
            seen.add(endpoint_pair)
        return self


class MinCostFlowRequest(StrictModel):
    graph: CostedFlowGraph
    demands: tuple[int, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.demands) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.demands_length_must_match_vertex_count",
                "demands length must match vertex_count",
            )
        if sum(self.demands) != 0:
            raise PydanticCustomError(
                "graph.demands_must_sum_to_zero", "demands must sum to zero"
            )
        capacity_denominators = tuple(
            edge.capacity.as_integer_ratio()[1] for edge in self.graph.edges
        )
        cost_denominators = tuple(
            edge.cost.as_integer_ratio()[1] for edge in self.graph.edges
        )
        _bounded_denominator_scale(capacity_denominators, "capacity")
        _bounded_denominator_scale(cost_denominators, "cost")
        return self


class FlowEdgeResult(StrictModel):
    """The flow assigned to one directed edge."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    flow: CanonicalRational


class MinCostFlowResult(StrictModel):
    """The exact minimum-cost-flow outcome bound to its complete source network.

    Retains the source graph and demands so validation replays the defining
    relations instead of trusting an independently authored claim: every edge
    flow lies between zero and its source capacity, every node balance equals
    its source demand, and the objective is the exact cost of the returned
    source-unit flows.  Infeasibility carries no flow or cost data.
    """

    graph: CostedFlowGraph
    demands: tuple[int, ...] = Field(default=(), max_length=64)
    total_cost: CanonicalRational
    flow_edges: tuple[FlowEdgeResult, ...] = Field(default=())
    feasible: bool
    convention: Literal["NETWORKX_MIN_COST_FLOW"] = "NETWORKX_MIN_COST_FLOW"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from fractions import Fraction

        if len(self.demands) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.demands_length_must_match_graph_vertex_count",
                "demands length must match graph.vertex_count",
            )
        if sum(self.demands) != 0:
            raise PydanticCustomError(
                "graph.demands_must_sum_to_zero", "demands must sum to zero"
            )
        if not self.feasible:
            if self.flow_edges or self.total_cost.as_fraction() != 0:
                raise PydanticCustomError(
                    "graph.infeasible_result_carries_no_flow_edges_nonzero",
                    "an infeasible result carries no flow edges or nonzero cost",
                )
            return self

        capacities: dict[tuple[int, int], Fraction] = {}
        costs: dict[tuple[int, int], Fraction] = {}
        for edge in self.graph.edges:
            endpoints = (edge.source, edge.target)
            capacities[endpoints] = edge.capacity.as_fraction()
            costs[endpoints] = edge.cost.as_fraction()
        balance = [Fraction(0)] * self.graph.vertex_count
        objective = Fraction(0)
        seen: set[tuple[int, int]] = set()
        for flow_edge in self.flow_edges:
            endpoints = (flow_edge.source, flow_edge.target)
            if endpoints not in capacities:
                raise PydanticCustomError(
                    "graph.flow_reported_undeclared_edge_endpoints_endpoints",
                    f"flow reported on undeclared edge {endpoints[0]}->{endpoints[1]}",
                )
            if endpoints in seen:
                raise PydanticCustomError(
                    "graph.edge_endpoints_endpoints_reported_more_than_once",
                    f"edge {endpoints[0]}->{endpoints[1]} reported more than once",
                )
            seen.add(endpoints)
            flow = flow_edge.flow.as_fraction()
            if not 0 <= flow <= capacities[endpoints]:
                raise PydanticCustomError(
                    "graph.flow_flow_edge_endpoints_endpoints_violates_source",
                    f"flow {flow} on edge {endpoints[0]}->{endpoints[1]} "
                    f"violates the source capacity {capacities[endpoints]}",
                )
            balance[flow_edge.source] -= flow
            balance[flow_edge.target] += flow
            objective += costs[endpoints] * flow
        for node, demand in enumerate(self.demands):
            if balance[node] != demand:
                raise PydanticCustomError(
                    "graph.node_node_balance_balance_node_does_equal",
                    f"node {node} balance {balance[node]} does not equal "
                    f"its source demand {demand}",
                )
        if objective != self.total_cost.as_fraction():
            raise PydanticCustomError(
                "graph.total_cost_does_not_equal_the_cost_of_the_return",
                "total cost does not equal the cost of the returned flows",
            )
        return self


class CirculationResult(StrictModel):
    feasible: bool
    flow_edges: tuple[FlowEdgeResult, ...] = Field(default=())
    convention: Literal["NETWORKX_CIRCULATION"] = "NETWORKX_CIRCULATION"
