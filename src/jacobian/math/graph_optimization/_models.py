"""Typed contracts for bounded finite-graph optimization."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.graph_optimization._coloring_models import (
    ChromaticGraph,
    GraphVertex,
)

OptimizationStatus = Literal["EXACT", "UNKNOWN"]
OptimizationTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "BOUND_CONVERGENCE",
    "WALL_TIME",
    "SOLVER_CALL_LIMIT",
    "SOLVER_UNKNOWN",
    "SPECIAL_CASE",
]
MAX_GRAPH_WEIGHT_DIGITS = 256


class RationalWeightedEdge(StrictModel):
    """One exact rational edge of a bounded simple undirected graph."""

    endpoints: tuple[GraphVertex, GraphVertex]
    weight: CanonicalRational

    @model_validator(mode="after")
    def require_distinct_endpoints_and_bounded_weight(self) -> Self:
        if self.endpoints[0] == self.endpoints[1]:
            raise ValueError("weighted graph edges must not contain self-loops")
        require_bounded_rational(
            self.weight,
            max_digits=MAX_GRAPH_WEIGHT_DIGITS,
            label="graph weight",
        )
        return self


class RationalWeightedGraph(StrictModel):
    """A bounded labelled simple graph with one exact rational edge weight."""

    weighted_graph_schema_version: Literal["1"] = "1"
    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    edges: tuple[RationalWeightedEdge, ...] = Field(max_length=496)

    @model_validator(mode="after")
    def require_simple_weighted_graph(self) -> Self:
        vertex_set = set(self.vertices)
        if len(vertex_set) != len(self.vertices):
            raise ValueError("weighted graph vertices must be unique")
        normalized_edges = {tuple(sorted(edge.endpoints)) for edge in self.edges}
        if any(
            endpoint not in vertex_set
            for edge in self.edges
            for endpoint in edge.endpoints
        ):
            raise ValueError("weighted graph edges must reference declared vertices")
        if len(normalized_edges) != len(self.edges):
            raise ValueError("weighted graph edges must be unique ignoring orientation")
        return self


class GraphMinimumSpanningTreeRequest(StrictModel):
    """One complete exact minimum-spanning-tree request."""

    graph: RationalWeightedGraph


class CanonicalWeightedTreeEdge(StrictModel):
    """One canonically oriented source edge selected for a spanning tree."""

    endpoints: tuple[GraphVertex, GraphVertex]
    weight: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_edge(self) -> Self:
        if self.endpoints[0] >= self.endpoints[1]:
            raise ValueError(
                "tree edge endpoints must be in strict lexicographic order"
            )
        require_bounded_rational(
            self.weight,
            max_digits=MAX_GRAPH_WEIGHT_DIGITS,
            label="graph weight",
        )
        return self


class GraphMstCycleCheck(StrictModel):
    """One non-tree edge's fundamental-cycle non-improvement check."""

    non_tree_edge: tuple[GraphVertex, GraphVertex]
    edge_weight: CanonicalRational
    tree_path_vertices: tuple[GraphVertex, ...] = Field(
        min_length=2,
        max_length=32,
    )
    maximum_tree_path_weight: CanonicalRational
    condition: Literal["EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT"] = (
        "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT"
    )

    @model_validator(mode="after")
    def require_canonical_cycle_check(self) -> Self:
        if self.non_tree_edge[0] >= self.non_tree_edge[1]:
            raise ValueError(
                "non-tree edge endpoints must be in strict lexicographic order"
            )
        if (
            self.tree_path_vertices[0] != self.non_tree_edge[0]
            or self.tree_path_vertices[-1] != self.non_tree_edge[1]
            or len(set(self.tree_path_vertices)) != len(self.tree_path_vertices)
        ):
            raise ValueError(
                "tree path must be simple and join the non-tree edge endpoints"
            )
        for weight in (self.edge_weight, self.maximum_tree_path_weight):
            require_bounded_rational(
                weight,
                max_digits=MAX_GRAPH_WEIGHT_DIGITS,
                label="graph weight",
            )
        return self


class GraphMstOptimalityCertificate(StrictModel):
    """Inspectable cycle-property certificate for one selected tree."""

    certificate_schema_version: Literal["1"] = "1"
    method: Literal["ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING"] = (
        "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING"
    )
    checks: tuple[GraphMstCycleCheck, ...] = Field(max_length=496)
    required_checks: tuple[
        Literal[
            "SOURCE_CONNECTIVITY",
            "TREE_SPANNING_ACYCLIC",
            "TOTAL_WEIGHT_EXACT",
            "ALL_NON_TREE_EDGES_COVERED",
            "CYCLE_NON_IMPROVEMENT",
        ],
        ...,
    ] = (
        "SOURCE_CONNECTIVITY",
        "TREE_SPANNING_ACYCLIC",
        "TOTAL_WEIGHT_EXACT",
        "ALL_NON_TREE_EDGES_COVERED",
        "CYCLE_NON_IMPROVEMENT",
    )

    @model_validator(mode="after")
    def require_canonical_check_order(self) -> Self:
        edges = tuple(check.non_tree_edge for check in self.checks)
        if edges != tuple(sorted(edges)) or len(edges) != len(set(edges)):
            raise ValueError(
                "cycle checks must cover unique canonically sorted non-tree edges"
            )
        return self


class GraphMinimumSpanningTreeResult(StrictModel):
    """Complete weighted spanning-tree outcome on the supplied finite graph."""

    result_schema_version: Literal["1"] = "1"
    status: Literal["EXACT", "NO_SPANNING_TREE"]
    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    order: StrictInt = Field(ge=0, le=32)
    connected: bool
    component_count: StrictInt = Field(ge=0, le=32)
    components: tuple[tuple[GraphVertex, ...], ...] = Field(max_length=32)
    tree_edges: tuple[CanonicalWeightedTreeEdge, ...] = Field(max_length=31)
    total_weight: CanonicalRational | None = None
    optimality_certificate: GraphMstOptimalityCertificate
    convention: Literal[
        "MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE"
    ] = "MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE"
    completion: Literal["COMPLETE"] = "COMPLETE"

    @model_validator(mode="after")
    def require_canonical_partition(self) -> Self:
        if (
            self.vertices != tuple(sorted(self.vertices))
            or len(self.vertices) != len(set(self.vertices))
            or self.order != len(self.vertices)
        ):
            raise ValueError("result vertices must be unique and canonically sorted")
        if self.component_count != len(self.components):
            raise ValueError("component count must match the component partition")
        if any(
            not component
            or component != tuple(sorted(component))
            or len(component) != len(set(component))
            for component in self.components
        ):
            raise ValueError(
                "components must be nonempty sets in canonical vertex order"
            )
        if self.components != tuple(
            sorted(self.components, key=lambda component: component[0])
        ):
            raise ValueError("components must be canonically ordered")
        partition = tuple(
            sorted(vertex for component in self.components for vertex in component)
        )
        if partition != self.vertices:
            raise ValueError("components must partition the result vertices")
        return self

    @model_validator(mode="after")
    def bind_tree_and_status(self) -> Self:
        tree_endpoints = tuple(edge.endpoints for edge in self.tree_edges)
        if tree_endpoints != tuple(sorted(tree_endpoints)) or len(
            tree_endpoints
        ) != len(set(tree_endpoints)):
            raise ValueError("tree edges must be unique and canonically sorted")
        if any(
            endpoint not in set(self.vertices)
            for edge in self.tree_edges
            for endpoint in edge.endpoints
        ):
            raise ValueError("tree edges must reference result vertices")
        if self.status == "EXACT":
            if (
                not self.vertices
                or not self.connected
                or self.component_count != 1
                or len(self.tree_edges) != self.order - 1
                or self.total_weight is None
            ):
                raise ValueError(
                    "exact MST result requires a connected nonempty spanning tree"
                )
        elif (
            self.connected
            or self.component_count == 1
            or self.tree_edges
            or self.total_weight is not None
            or self.optimality_certificate.checks
        ):
            raise ValueError(
                "no-spanning-tree result must expose only the disconnected partition"
            )
        return self


class GraphOptimizationBudget(StrictModel):
    """Explicit size, solver-call, and wall-clock limits."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(default=33, ge=1, le=33)
    max_order: StrictInt = Field(default=32, ge=0, le=32)


class GraphOptimizationRequest(StrictModel):
    """One bounded simple-undirected-graph optimization request."""

    graph: ChromaticGraph
    resource_budget: GraphOptimizationBudget = Field(
        default_factory=GraphOptimizationBudget
    )

    @model_validator(mode="after")
    def enforce_order_budget(self) -> Self:
        if len(self.graph.vertices) > self.resource_budget.max_order:
            raise ValueError("graph order exceeds the declared max_order budget")
        return self


class GraphHamiltonianPathRequest(StrictModel):
    """One finite simple graph inside the complete decision/checker scope."""

    graph: ChromaticGraph

    @model_validator(mode="after")
    def enforce_complete_decision_scope(self) -> Self:
        if len(self.graph.vertices) > 18:
            raise ValueError(
                "Hamiltonian-path decision supports graphs of order at most 18"
            )
        return self


class GraphHamiltonianPathResult(StrictModel):
    """Complete spanning simple-path decision on the supplied finite graph."""

    result_schema_version: Literal["1"] = "1"
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    order: StrictInt = Field(ge=0, le=18)
    path: tuple[GraphVertex, ...] = Field(max_length=18)
    convention: Literal["EMPTY_GRAPH_HAS_EMPTY_HAMILTONIAN_PATH"] = (
        "EMPTY_GRAPH_HAS_EMPTY_HAMILTONIAN_PATH"
    )
    completion: Literal["COMPLETE"] = "COMPLETE"

    @model_validator(mode="after")
    def bind_decision_and_path(self) -> Self:
        if len(set(self.path)) != len(self.path):
            raise ValueError("Hamiltonian path vertices must be unique")
        if self.decision == "EXISTS":
            if len(self.path) != self.order:
                raise ValueError("EXISTS requires one spanning path witness")
        elif self.path:
            raise ValueError("DOES_NOT_EXIST must not carry a path witness")
        return self


class OptimizationSearchStep(StrictModel):
    """One threshold-feasibility decision."""

    bound: StrictInt = Field(ge=0, le=32)
    relation: Literal["AT_MOST", "AT_LEAST"]
    status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]


class _OptimizationOutput(StrictModel):
    status: OptimizationStatus
    order: StrictInt = Field(ge=0, le=32)
    optimum_value: StrictInt | None = Field(default=None, ge=0, le=32)
    incumbent_value: StrictInt = Field(ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    tested: tuple[OptimizationSearchStep, ...]
    termination_reason: OptimizationTermination
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_status_and_bounds(self) -> Self:
        if not self.lower_bound <= self.incumbent_value <= self.upper_bound:
            raise ValueError("incumbent must lie within the reported bounds")
        if self.status == "EXACT":
            if (
                self.optimum_value is None
                or self.lower_bound != self.optimum_value
                or self.upper_bound != self.optimum_value
                or self.incumbent_value != self.optimum_value
            ):
                raise ValueError("exact result must have one coincident optimum")
        elif self.optimum_value is not None:
            raise ValueError("unknown result cannot claim an optimum")
        return self


class _VertexOptimizationOutput(_OptimizationOutput):
    witness_vertices: tuple[GraphVertex, ...]

    @model_validator(mode="after")
    def bind_vertex_witness(self) -> Self:
        if len(set(self.witness_vertices)) != len(self.witness_vertices):
            raise ValueError("witness vertices must be unique")
        if tuple(sorted(self.witness_vertices)) != self.witness_vertices:
            raise ValueError("witness vertices must be canonically sorted")
        if len(self.witness_vertices) != self.incumbent_value:
            raise ValueError("vertex witness cardinality must match the incumbent")
        return self


class GraphDominationMinimumOutput(_VertexOptimizationOutput):
    """Minimum ordinary closed-neighborhood dominating-set result."""

    convention: Literal["ORDINARY_CLOSED_NEIGHBORHOOD"] = "ORDINARY_CLOSED_NEIGHBORHOOD"

    @model_validator(mode="after")
    def bind_minimum_incumbent(self) -> Self:
        if self.incumbent_value != self.upper_bound:
            raise ValueError("a minimum-search incumbent is an upper bound")
        return self


class GraphMinimumMaximalMatchingOutput(_OptimizationOutput):
    """Minimum-cardinality maximal matching (saturation number)."""

    convention: Literal["MINIMUM_CARDINALITY_MAXIMAL_MATCHING"] = (
        "MINIMUM_CARDINALITY_MAXIMAL_MATCHING"
    )
    witness_edges: tuple[tuple[GraphVertex, GraphVertex], ...]

    @model_validator(mode="after")
    def bind_matching_witness(self) -> Self:
        if self.incumbent_value != self.upper_bound:
            raise ValueError("a minimum-search incumbent is an upper bound")
        if len(self.witness_edges) != self.incumbent_value:
            raise ValueError("matching witness cardinality must match the incumbent")
        if (
            any(left >= right for left, right in self.witness_edges)
            or len(set(self.witness_edges)) != len(self.witness_edges)
            or tuple(sorted(self.witness_edges)) != self.witness_edges
        ):
            raise ValueError("matching edges must be unique and canonically sorted")
        return self


class _MaximumVertexOptimizationOutput(_VertexOptimizationOutput):
    @model_validator(mode="after")
    def bind_maximum_incumbent(self) -> Self:
        if self.incumbent_value != self.lower_bound:
            raise ValueError("a maximum-search incumbent is a lower bound")
        return self


class GraphInducedForestMaximumOutput(_MaximumVertexOptimizationOutput):
    convention: Literal["EMPTY_ALLOWED_ACYCLIC_INDUCED_SUBGRAPH"] = (
        "EMPTY_ALLOWED_ACYCLIC_INDUCED_SUBGRAPH"
    )


class GraphInducedTreeMaximumOutput(_MaximumVertexOptimizationOutput):
    convention: Literal["NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"] = (
        "NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"
    )


class GraphInducedBipartiteMaximumOutput(_MaximumVertexOptimizationOutput):
    convention: Literal["EMPTY_ALLOWED_TWO_COLORABLE_INDUCED_SUBGRAPH"] = (
        "EMPTY_ALLOWED_TWO_COLORABLE_INDUCED_SUBGRAPH"
    )


class _VertexOptimalityObligation(StrictModel):
    graph: ChromaticGraph
    status: OptimizationStatus
    claimed_value: StrictInt | None = Field(default=None, ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    witness_vertices: tuple[GraphVertex, ...]
    tested: tuple[OptimizationSearchStep, ...]

    @model_validator(mode="after")
    def bind_claim(self) -> Self:
        if self.status == "EXACT" and (
            self.claimed_value is None
            or self.lower_bound != self.claimed_value
            or self.upper_bound != self.claimed_value
        ):
            raise ValueError("exact obligation must bind the optimum")
        if self.status == "UNKNOWN" and self.claimed_value is not None:
            raise ValueError("incomplete search cannot claim an optimum")
        return self


class GraphDominationMinimumObligation(_VertexOptimalityObligation):
    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_DOMINATION_MINIMUM_OPTIMALITY"] = (
        "GRAPH_DOMINATION_MINIMUM_OPTIMALITY"
    )
    convention: Literal["ORDINARY_CLOSED_NEIGHBORHOOD"] = "ORDINARY_CLOSED_NEIGHBORHOOD"
    required_checks: tuple[
        Literal["DOMINATING_SET_FEASIBILITY", "MINIMUM_CARDINALITY"],
        ...,
    ] = ("DOMINATING_SET_FEASIBILITY", "MINIMUM_CARDINALITY")


class GraphInducedForestMaximumObligation(_VertexOptimalityObligation):
    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_INDUCED_FOREST_MAXIMUM_OPTIMALITY"] = (
        "GRAPH_INDUCED_FOREST_MAXIMUM_OPTIMALITY"
    )
    convention: Literal["EMPTY_ALLOWED_ACYCLIC_INDUCED_SUBGRAPH"] = (
        "EMPTY_ALLOWED_ACYCLIC_INDUCED_SUBGRAPH"
    )
    required_checks: tuple[
        Literal["INDUCED_FOREST_FEASIBILITY", "MAXIMUM_CARDINALITY"],
        ...,
    ] = ("INDUCED_FOREST_FEASIBILITY", "MAXIMUM_CARDINALITY")


class GraphInducedTreeMaximumObligation(_VertexOptimalityObligation):
    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_INDUCED_TREE_MAXIMUM_OPTIMALITY"] = (
        "GRAPH_INDUCED_TREE_MAXIMUM_OPTIMALITY"
    )
    convention: Literal["NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"] = (
        "NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"
    )
    required_checks: tuple[
        Literal["INDUCED_TREE_FEASIBILITY", "MAXIMUM_CARDINALITY"],
        ...,
    ] = ("INDUCED_TREE_FEASIBILITY", "MAXIMUM_CARDINALITY")


class GraphInducedBipartiteMaximumObligation(_VertexOptimalityObligation):
    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_INDUCED_BIPARTITE_MAXIMUM_OPTIMALITY"] = (
        "GRAPH_INDUCED_BIPARTITE_MAXIMUM_OPTIMALITY"
    )
    convention: Literal["EMPTY_ALLOWED_TWO_COLORABLE_INDUCED_SUBGRAPH"] = (
        "EMPTY_ALLOWED_TWO_COLORABLE_INDUCED_SUBGRAPH"
    )
    required_checks: tuple[
        Literal["INDUCED_BIPARTITE_FEASIBILITY", "MAXIMUM_CARDINALITY"],
        ...,
    ] = ("INDUCED_BIPARTITE_FEASIBILITY", "MAXIMUM_CARDINALITY")


class GraphMinimumMaximalMatchingObligation(StrictModel):
    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["GRAPH_MINIMUM_MAXIMAL_MATCHING_OPTIMALITY"] = (
        "GRAPH_MINIMUM_MAXIMAL_MATCHING_OPTIMALITY"
    )
    convention: Literal["MINIMUM_CARDINALITY_MAXIMAL_MATCHING"] = (
        "MINIMUM_CARDINALITY_MAXIMAL_MATCHING"
    )
    graph: ChromaticGraph
    status: OptimizationStatus
    claimed_value: StrictInt | None = Field(default=None, ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    witness_edges: tuple[tuple[GraphVertex, GraphVertex], ...]
    tested: tuple[OptimizationSearchStep, ...]
    required_checks: tuple[
        Literal["MAXIMAL_MATCHING_FEASIBILITY", "MINIMUM_CARDINALITY"],
        ...,
    ] = ("MAXIMAL_MATCHING_FEASIBILITY", "MINIMUM_CARDINALITY")

    @model_validator(mode="after")
    def bind_claim(self) -> Self:
        if self.status == "EXACT" and (
            self.claimed_value is None
            or self.lower_bound != self.claimed_value
            or self.upper_bound != self.claimed_value
        ):
            raise ValueError("exact obligation must bind the saturation number")
        if self.status == "UNKNOWN" and self.claimed_value is not None:
            raise ValueError("incomplete search cannot claim an optimum")
        return self
