"""Typed wire contracts for graph flow and cut operations."""

from __future__ import annotations

from math import lcm
from typing import Self

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


class MinCutRequest(StrictModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)


class MinCutResult(StrictModel):
    cut_value: CanonicalRational
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]


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


class EdgeDisjointPathsResult(StrictModel):
    path_count: int = Field(ge=0)
    paths: tuple[tuple[int, ...], ...] = Field(default=())
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)


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


class FlowEdgeResult(StrictModel):
    """The flow assigned to one directed edge."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    flow: CanonicalRational


class MinCostFlowResult(StrictModel):
    """The exact minimum-cost-flow outcome bound to its source network.

    The producer establishes feasibility, conservation, capacities, and the
    objective once. Parsing retains only the result's structural shape;
    deliberate verification of an independently supplied claim belongs to
    the flow owner.
    """

    graph: CostedFlowGraph
    demands: tuple[int, ...] = Field(default=(), max_length=64)
    total_cost: CanonicalRational
    flow_edges: tuple[FlowEdgeResult, ...] = Field(default=())
    feasible: bool

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if len(self.demands) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.demands_length_must_match_graph_vertex_count",
                "demands length must match graph.vertex_count",
            )
        if not self.feasible and (
            self.flow_edges or self.total_cost.as_fraction() != 0
        ):
            raise PydanticCustomError(
                "graph.infeasible_result_carries_no_flow_edges_nonzero",
                "an infeasible result carries no flow edges or nonzero cost",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: MinCostFlowRequest,
        *,
        total_cost: CanonicalRational,
        feasible: bool,
        flow_edges: tuple[FlowEdgeResult, ...],
    ) -> Self:
        """Build one result after the admitted flow kernel established it."""

        return cls.model_construct(
            graph=request.graph,
            demands=request.demands,
            total_cost=total_cost,
            feasible=feasible,
            flow_edges=flow_edges,
        )


class CirculationResult(StrictModel):
    feasible: bool
    flow_edges: tuple[FlowEdgeResult, ...] = Field(default=())
