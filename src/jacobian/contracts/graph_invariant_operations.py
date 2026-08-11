"""Typed contracts for finite simple-graph invariants."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.graph_coloring import ChromaticGraph, GraphVertex
from jacobian.contracts.graph_optimization import (
    OptimizationSearchStep,
    OptimizationStatus,
    OptimizationTermination,
)
from jacobian.contracts.results import ContractModel


class GraphInvariantRequest(ContractModel):
    graph: ChromaticGraph


GraphDistance = Annotated[StrictInt, Field(ge=0, le=31)] | None
GraphDistanceRow = Annotated[
    tuple[GraphDistance, ...],
    Field(max_length=32),
]


def _validate_distance_matrix_shape(
    vertices: tuple[GraphVertex, ...],
    distances: tuple[GraphDistanceRow, ...],
) -> int:
    order = len(vertices)
    if tuple(sorted(vertices)) != vertices or len(set(vertices)) != order:
        raise ValueError("distance-matrix vertices must be unique and sorted")
    if len(distances) != order or any(len(row) != order for row in distances):
        raise ValueError("distance matrix must be square on the declared vertices")
    return order


def _validate_distance_matrix_diagonal_and_symmetry(
    distances: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    for source in range(order):
        for target in range(order):
            distance = distances[source][target]
            if source == target:
                if distance != 0:
                    raise ValueError("distance-matrix diagonal must be zero")
            elif distance == 0:
                raise ValueError("off-diagonal distances must be positive or null")
            if distance != distances[target][source]:
                raise ValueError("undirected distance matrix must be symmetric")


def _validate_distance_matrix_triangle_inequality(
    distances: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    for source in range(order):
        for intermediate in range(order):
            left = distances[source][intermediate]
            if left is None:
                continue
            for target in range(order):
                right = distances[intermediate][target]
                if right is None:
                    continue
                direct = distances[source][target]
                if direct is None or direct > left + right:
                    raise ValueError(
                        "finite distances must satisfy component closure and "
                        "the triangle inequality"
                    )


class GraphDistanceMatrixResult(ContractModel):
    """All exact unweighted shortest-path distances in canonical vertex order."""

    semantics_version: Literal["unweighted-shortest-path-distance-matrix.v1"]
    vertex_ordering: Literal["LEXICOGRAPHIC_ASCENDING"]
    pair_coverage: Literal["ALL_ORDERED_VERTEX_PAIRS"]
    unreachable_representation: Literal["JSON_NULL"]
    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    distances: tuple[GraphDistanceRow, ...] = Field(max_length=32)
    connected: StrictBool

    @model_validator(mode="after")
    def bind_complete_metric(self) -> Self:
        order = _validate_distance_matrix_shape(self.vertices, self.distances)
        _validate_distance_matrix_diagonal_and_symmetry(self.distances, order)
        _validate_distance_matrix_triangle_inequality(self.distances, order)
        expected_connected = order > 0 and all(
            distance is not None for row in self.distances for distance in row
        )
        if self.connected != expected_connected:
            raise ValueError("connected must match all-pairs finite reachability")
        return self


class GraphGirthResult(ContractModel):
    girth: StrictInt = Field(ge=0, le=32)
    has_cycle: StrictBool

    @model_validator(mode="after")
    def bind_cycle_status(self) -> Self:
        if self.has_cycle != (self.girth > 0):
            raise ValueError("has_cycle must agree with the girth sentinel")
        return self


class GraphDiameterResult(ContractModel):
    status: Literal["COMPUTED", "NOT_APPLICABLE"]
    diameter: StrictInt | None = Field(default=None, ge=0, le=31)
    connected: StrictBool
    exactness: Literal["EXACT", "NOT_APPLICABLE"]
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_connectivity(self) -> Self:
        if self.status == "COMPUTED":
            if (
                self.diameter is None
                or not self.connected
                or self.exactness != "EXACT"
                or self.detail is not None
            ):
                raise ValueError(
                    "computed diameter requires an exact value on a connected graph"
                )
        elif (
            self.diameter is not None
            or self.connected
            or self.exactness != "NOT_APPLICABLE"
            or self.detail is None
        ):
            raise ValueError(
                "inapplicable diameter requires no value and an explicit detail"
            )
        return self


class GraphEdgeConnectivityResult(ContractModel):
    edge_connectivity: StrictInt = Field(ge=0, le=31)


class GraphVertexConnectivityResult(ContractModel):
    vertex_connectivity: StrictInt = Field(ge=0, le=31)


class GraphEulerianResult(ContractModel):
    is_eulerian: StrictBool


class GraphSpanningTreeCountResult(ContractModel):
    spanning_tree_count: StrictInt = Field(ge=0)
    connected: StrictBool


class GraphTutteBergeCertificate(ContractModel):
    certificate_schema_version: Literal["1"] = "1"
    kind: Literal["TUTTE_BERGE_BARRIER"] = "TUTTE_BERGE_BARRIER"
    barrier_vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    odd_component_count: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=16)

    @model_validator(mode="after")
    def require_canonical_barrier(self) -> Self:
        if tuple(sorted(self.barrier_vertices)) != self.barrier_vertices or len(
            set(self.barrier_vertices)
        ) != len(self.barrier_vertices):
            raise ValueError("Tutte-Berge barrier vertices must be unique and sorted")
        return self


class GraphMaximumMatchingResult(ContractModel):
    maximum_matching_cardinality: StrictInt = Field(ge=0, le=16)
    witness_edges: tuple[tuple[GraphVertex, GraphVertex], ...]
    certificate: GraphTutteBergeCertificate

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if len(self.witness_edges) != self.maximum_matching_cardinality:
            raise ValueError("matching witness cardinality must match the result")
        if self.certificate.upper_bound != self.maximum_matching_cardinality:
            raise ValueError("Tutte-Berge upper bound must match the result")
        if (
            any(left >= right for left, right in self.witness_edges)
            or tuple(sorted(self.witness_edges)) != self.witness_edges
            or len({vertex for edge in self.witness_edges for vertex in edge})
            != 2 * len(self.witness_edges)
        ):
            raise ValueError("matching witness must be canonical and vertex-disjoint")
        return self


class GraphTriangleCountResult(ContractModel):
    triangle_count: StrictInt = Field(ge=0, le=4_960)


class GraphCoreRequest(GraphInvariantRequest):
    k: StrictInt = Field(ge=0, le=32)


class GraphCoreResult(ContractModel):
    k: StrictInt = Field(ge=0, le=32)
    vertices: tuple[GraphVertex, ...]

    @model_validator(mode="after")
    def require_canonical_vertices(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices:
            raise ValueError("k-core vertices must be canonically sorted")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("k-core vertices must be unique")
        return self


class GraphRadiusResult(ContractModel):
    status: Literal["COMPUTED", "NOT_APPLICABLE"]
    radius: StrictInt | None = Field(default=None, ge=0, le=31)
    connected: StrictBool
    exactness: Literal["EXACT", "NOT_APPLICABLE"]
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_connectivity(self) -> Self:
        if self.status == "COMPUTED":
            if (
                self.radius is None
                or not self.connected
                or self.exactness != "EXACT"
                or self.detail is not None
            ):
                raise ValueError(
                    "computed radius requires an exact value on a connected graph"
                )
        elif (
            self.radius is not None
            or self.connected
            or self.exactness != "NOT_APPLICABLE"
            or self.detail is None
        ):
            raise ValueError(
                "inapplicable radius requires no value and an explicit detail"
            )
        return self


class GraphCardinalityMaximumResult(ContractModel):
    status: OptimizationStatus
    order: StrictInt = Field(ge=0, le=32)
    optimum_value: StrictInt | None = Field(default=None, ge=0, le=32)
    incumbent_value: StrictInt = Field(ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    witness_vertices: tuple[GraphVertex, ...]
    tested: tuple[OptimizationSearchStep, ...]
    termination_reason: OptimizationTermination
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_claim_and_witness(self) -> Self:
        if self.incumbent_value != len(self.witness_vertices):
            raise ValueError("witness cardinality must match the incumbent")
        if tuple(sorted(self.witness_vertices)) != self.witness_vertices:
            raise ValueError("witness vertices must be canonically sorted")
        if self.lower_bound != self.incumbent_value:
            raise ValueError("a maximum-search incumbent is the lower bound")
        if self.status == "EXACT":
            if (
                self.optimum_value is None
                or self.lower_bound != self.optimum_value
                or self.upper_bound != self.optimum_value
            ):
                raise ValueError("exact result must bind one coincident optimum")
        elif self.optimum_value is not None:
            raise ValueError("incomplete search cannot claim an optimum")
        return self


class GraphCliqueNumberResult(GraphCardinalityMaximumResult):
    convention: Literal["MAXIMUM_COMPLETE_VERTEX_SUBSET"] = (
        "MAXIMUM_COMPLETE_VERTEX_SUBSET"
    )


class GraphIndependenceNumberResult(GraphCardinalityMaximumResult):
    convention: Literal["MAXIMUM_EDGE_FREE_VERTEX_SUBSET"] = (
        "MAXIMUM_EDGE_FREE_VERTEX_SUBSET"
    )


class GraphCardinalityMaximumObligation(ContractModel):
    obligation_schema_version: Literal["1"] = "1"
    graph: ChromaticGraph
    predicate: Literal[
        "GRAPH_CLIQUE_NUMBER_OPTIMALITY",
        "GRAPH_INDEPENDENCE_NUMBER_OPTIMALITY",
    ]
    status: OptimizationStatus
    claimed_value: StrictInt | None = Field(default=None, ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    witness_vertices: tuple[GraphVertex, ...]
    tested: tuple[OptimizationSearchStep, ...]
    required_checks: tuple[
        Literal["WITNESS_FEASIBILITY", "MAXIMUM_CARDINALITY"],
        ...,
    ] = ("WITNESS_FEASIBILITY", "MAXIMUM_CARDINALITY")
