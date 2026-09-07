"""Typed wire contracts for graph flow and cut operations."""

from __future__ import annotations

from math import lcm
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

# Derived integer scales that make rational capacities and costs exact
# integers are intermediate growth, not input size: each denominator is
# already bounded by the canonical rational limit, but the least common
# multiple of up to 1,024 such denominators can grow far beyond what the
# integer backend can expand. Requests whose derived scale exceeds this
# documented conservative digit budget are rejected before any backend graph
# is constructed.
MAX_MIN_COST_FLOW_DERIVED_SCALE_DIGITS = 4096
MAX_BIPARTITE_FACTOR_DEMAND = 1_000_000_000
MAX_BIPARTITE_FACTOR_DEMAND_DIGITS = 9
MAX_BIPARTITE_FACTOR_ARCS = 512


BipartiteFactorStatus = Literal["FOUND", "INFEASIBLE"]


def _bounded_denominator_scale(denominators: tuple[int, ...], kind: str) -> int:
    """Return the LCM of ``denominators`` under the derived-scale digit budget."""
    scale = 1
    for denominator in denominators:
        scale = lcm(scale, abs(denominator))
        if (
            len(format_canonical_integer(scale))
            > MAX_MIN_COST_FLOW_DERIVED_SCALE_DIGITS
        ):
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


class MaxFlowResult(MaxFlowRequest):
    flow_value: CanonicalRational
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    flow_edges: tuple[FlowEdgeValue, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_edge_axis(self) -> Self:
        source_edges = {(edge.source, edge.target) for edge in self.graph.edges}
        edges = tuple((edge.source, edge.target) for edge in self.flow_edges)
        if len(set(edges)) != len(edges) or not set(edges) <= source_edges:
            raise ValueError("flow edges must be distinct edges of the source network")
        return self


class MinCutRequest(StrictModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)


class MinCutResult(MinCutRequest):
    cut_value: CanonicalRational
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]

    @model_validator(mode="after")
    def require_partition_axis(self) -> Self:
        if sorted((*self.reachable, *self.unreachable)) != list(
            range(self.graph.vertex_count)
        ):
            raise ValueError(
                "cut sides must partition the retained network vertex axis"
            )
        return self


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


class EdgeDisjointPathsResult(EdgeDisjointPathsRequest):
    path_count: int = Field(ge=0)
    paths: tuple[tuple[int, ...], ...] = Field(default=(), max_length=512)
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)

    @model_validator(mode="after")
    def require_path_axes(self) -> Self:
        if any(
            not 2 <= len(path) <= self.graph.vertex_count
            or any(not 0 <= vertex < self.graph.vertex_count for vertex in path)
            for path in self.paths
        ):
            raise ValueError("paths must use the retained network vertex axis")
        return self


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


class BipartiteDegreeRequirements(StrictModel):
    """One nonnegative required degree per vertex of a source graph."""

    left: tuple[int, ...] = Field(min_length=1, max_length=64)
    right: tuple[int, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_nonnegative_bounded_degrees(self) -> Self:
        for side_name in ("left", "right"):
            side = getattr(self, side_name)
            for position, degree in enumerate(side):
                if degree < 0 or degree > MAX_BIPARTITE_FACTOR_DEMAND:
                    raise PydanticCustomError(
                        "graph.bipartite_factor_degree_must_be_in_0_max",
                        "every required degree must be in 0.."
                        f"{MAX_BIPARTITE_FACTOR_DEMAND}",
                        {"side": side_name, "position": position},
                    )
        return self


class BipartiteFactorRequest(StrictModel):
    """One bounded indexed bipartite graph plus complete degree requirements.

    The first slice admits at most 64 vertices per side, one nonnegative
    degree of at most 9 digits per source vertex, and at most 512 network
    edges. Disconnected graphs and isolated vertices remain valid.
    """

    graph: IndexedSimpleUndirectedGraph
    left: tuple[int, ...] = Field(min_length=1, max_length=64)
    right: tuple[int, ...] = Field(min_length=1, max_length=64)
    required_degrees: tuple[int, ...] = Field(
        min_length=2,
        max_length=128,
        description=(
            "One required degree per source vertex in graph-axis order: "
            "left vertices first, then right vertices."
        ),
    )

    @model_validator(mode="after")
    def require_source_bound_bipartition(self) -> Self:
        if tuple(sorted(self.left)) != tuple(range(len(self.left))):
            raise PydanticCustomError(
                "graph.bipartite_factor_left_axis_must_be_0_len",
                "left must be the complete ordered axis 0..len(left)-1",
            )
        if tuple(sorted(self.right)) != tuple(
            range(len(self.left), len(self.left) + len(self.right))
        ):
            raise PydanticCustomError(
                "graph.bipartite_factor_right_axis_must_follow_left",
                "right must be the complete ordered axis "
                "len(left)..len(left)+len(right)-1",
            )
        if len(self.left) + len(self.right) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.bipartite_factor_axes_must_partition_graph_vertices",
                "left and right must partition the graph vertex axis",
            )
        left_set = set(self.left)
        for edge in self.graph.edges:
            if (edge[0] in left_set) == (edge[1] in left_set):
                raise PydanticCustomError(
                    "graph.bipartite_factor_edges_must_cross_partition",
                    "every source edge must cross the supplied partition",
                )
        if len(self.required_degrees) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.bipartite_factor_requirements_must_cover_axes",
                "required_degrees must contain exactly one degree per source vertex",
            )
        if any(
            degree < 0 or degree > MAX_BIPARTITE_FACTOR_DEMAND
            for degree in self.required_degrees
        ):
            raise PydanticCustomError(
                "graph.bipartite_factor_degree_must_be_in_0_max",
                f"every required degree must be in 0..{MAX_BIPARTITE_FACTOR_DEMAND}",
            )
        if len(self.graph.edges) > MAX_BIPARTITE_FACTOR_ARCS:
            raise PydanticCustomError(
                "graph.bipartite_factor_edges_exceed_max",
                f"bipartite factor requests admit at most {MAX_BIPARTITE_FACTOR_ARCS} edges",
            )
        return self


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


class BipartiteFactorObstruction(StrictModel):
    """A replayable Hall obstruction in source vertex indices."""

    side: Literal["LEFT", "RIGHT"]
    vertices: tuple[int, ...] = Field(min_length=1, max_length=64)
    neighbors: tuple[int, ...] = Field(min_length=1, max_length=64)
    required: int = Field(ge=0, le=MAX_BIPARTITE_FACTOR_DEMAND)
    capacity: int = Field(ge=0, le=MAX_BIPARTITE_FACTOR_DEMAND * 64)

    @model_validator(mode="after")
    def require_neighbor_inequality(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices or len(
            set(self.vertices)
        ) != len(self.vertices):
            raise PydanticCustomError(
                "graph.bipartite_factor_vertices_must_be_unique_sorted",
                "obstruction vertices must be unique and increasing",
            )
        if tuple(sorted(self.neighbors)) != self.neighbors or len(
            set(self.neighbors)
        ) != len(self.neighbors):
            raise PydanticCustomError(
                "graph.bipartite_factor_neighbors_must_be_unique_sorted",
                "obstruction neighbors must be unique and increasing",
            )
        if self.required <= self.capacity:
            raise PydanticCustomError(
                "graph.bipartite_factor_obstruction_must_violate",
                "obstruction must satisfy required > neighbor capacity",
            )
        return self


class BipartiteFactorResult(StrictModel):
    """One exact spanning prescribed-degree factor or Hall obstruction."""

    graph: IndexedSimpleUndirectedGraph
    left: tuple[int, ...] = Field(min_length=1, max_length=64)
    right: tuple[int, ...] = Field(min_length=1, max_length=64)
    requirements: BipartiteDegreeRequirements
    status: BipartiteFactorStatus
    selected_edge_indices: tuple[int, ...] = Field(
        default=(), max_length=MAX_BIPARTITE_FACTOR_ARCS
    )
    obstruction: BipartiteFactorObstruction | None = None

    @model_validator(mode="after")
    def require_outcome_witness(self) -> Self:
        if self.status == "FOUND":
            if self.obstruction is not None:
                raise PydanticCustomError(
                    "graph.bipartite_factor_found_result_has_no_obstruction",
                    "a found factor must not carry an obstruction",
                )
            if len(set(self.selected_edge_indices)) != len(self.selected_edge_indices):
                raise PydanticCustomError(
                    "graph.bipartite_factor_selected_edges_must_be_distinct",
                    "selected edge indices must be distinct",
                )
            if any(
                not 0 <= index < len(self.graph.edges)
                for index in self.selected_edge_indices
            ):
                raise PydanticCustomError(
                    "graph.bipartite_factor_selected_edges_must_be_source_bound",
                    "selected edge indices must index the source graph",
                )
        else:
            if self.selected_edge_indices:
                raise PydanticCustomError(
                    "graph.bipartite_factor_infeasible_result_has_no_edges",
                    "an infeasible result must not carry selected edges",
                )
            if self.obstruction is None:
                raise PydanticCustomError(
                    "graph.bipartite_factor_infeasible_result_needs_obstruction",
                    "an infeasible result must carry an obstruction",
                )
        return self
