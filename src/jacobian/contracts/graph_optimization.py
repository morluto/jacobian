"""Typed contracts for bounded finite-graph optimization."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.graph_coloring import ChromaticGraph, GraphVertex
from jacobian.contracts.results import ContractModel

OptimizationStatus = Literal["EXACT", "UNKNOWN"]
OptimizationTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "BOUND_CONVERGENCE",
    "WALL_TIME",
    "SOLVER_CALL_LIMIT",
    "SOLVER_UNKNOWN",
    "SPECIAL_CASE",
]


class GraphOptimizationBudget(ContractModel):
    """Explicit size, solver-call, and wall-clock limits."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(default=33, ge=1, le=33)
    max_order: StrictInt = Field(default=32, ge=0, le=32)


class GraphOptimizationRequest(ContractModel):
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


class GraphHamiltonianPathRequest(ContractModel):
    """One finite simple graph inside the complete decision/checker scope."""

    graph: ChromaticGraph

    @model_validator(mode="after")
    def enforce_complete_decision_scope(self) -> Self:
        if len(self.graph.vertices) > 18:
            raise ValueError(
                "Hamiltonian-path decision supports graphs of order at most 18"
            )
        return self


class GraphHamiltonianPathResult(ContractModel):
    """Complete spanning simple-path decision on the supplied finite graph."""

    result_schema_version: Literal["1"] = "1"
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    order: StrictInt = Field(ge=0, le=18)
    path: tuple[GraphVertex, ...] = Field(max_length=18)
    convention: Literal["EMPTY_GRAPH_HAS_EMPTY_HAMILTONIAN_PATH"] = (
        "EMPTY_GRAPH_HAS_EMPTY_HAMILTONIAN_PATH"
    )
    completion: Literal["COMPLETE"] = "COMPLETE"
    verification_capability_id: Literal["graph.hamiltonian_path.verify"] = (
        "graph.hamiltonian_path.verify"
    )
    verification_input_field: Literal["result_uri"] = "result_uri"

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


class OptimizationSearchStep(ContractModel):
    """One threshold-feasibility decision."""

    bound: StrictInt = Field(ge=0, le=32)
    relation: Literal["AT_MOST", "AT_LEAST"]
    status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]


class _OptimizationOutput(ContractModel):
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


class _VertexOptimalityObligation(ContractModel):
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


class GraphMinimumMaximalMatchingObligation(ContractModel):
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
