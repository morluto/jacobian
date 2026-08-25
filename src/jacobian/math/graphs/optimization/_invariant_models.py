"""Typed contracts for finite simple-graph invariants."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._coloring_models import (
    ChromaticGraph,
    GraphVertex,
    PolynomialTimeGraph,
)
from jacobian.math.graphs.optimization._models import (
    OptimizationSearchStep,
    OptimizationStatus,
    OptimizationTermination,
)


class GraphInvariantRequest(StrictModel):
    graph: PolynomialTimeGraph


class GraphMaximumMatchingGraph(PolynomialTimeGraph):
    """A simple graph bounded for the polynomial-time matching operation."""

    vertices: tuple[GraphVertex, ...] = Field(max_length=256)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=32640)


class GraphMaximumMatchingRequest(StrictModel):
    graph: GraphMaximumMatchingGraph


class GraphGirthResult(StrictModel):
    girth: StrictInt = Field(ge=0, le=256)
    has_cycle: StrictBool

    @model_validator(mode="after")
    def bind_cycle_status(self) -> Self:
        if self.has_cycle != (self.girth > 0):
            raise PydanticCustomError(
                "graph.has_cycle_must_agree_with_the_girth_sentinel",
                "has_cycle must agree with the girth sentinel",
            )
        return self


class GraphDiameterResult(StrictModel):
    status: Literal["COMPUTED", "NOT_APPLICABLE"]
    diameter: StrictInt | None = Field(default=None, ge=0, le=255)
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
                raise PydanticCustomError(
                    "graph.computed_diameter_requires_exact_value_connected",
                    "computed diameter requires an exact value on a connected graph",
                )
        elif (
            self.diameter is not None
            or self.connected
            or self.exactness != "NOT_APPLICABLE"
            or self.detail is None
        ):
            raise PydanticCustomError(
                "graph.inapplicable_diameter_requires_no_value_explicit_detail",
                "inapplicable diameter requires no value and an explicit detail",
            )
        return self


class GraphEdgeConnectivityResult(StrictModel):
    edge_connectivity: StrictInt = Field(ge=0, le=255)


class GraphVertexConnectivityResult(StrictModel):
    vertex_connectivity: StrictInt = Field(ge=0, le=255)


class GraphEulerianResult(StrictModel):
    is_eulerian: StrictBool


class GraphSpanningTreeCountResult(StrictModel):
    spanning_tree_count: StrictInt = Field(ge=0)
    connected: StrictBool


class GraphTutteBergeCertificate(StrictModel):
    kind: Literal["TUTTE_BERGE_BARRIER"] = "TUTTE_BERGE_BARRIER"
    barrier_vertices: tuple[GraphVertex, ...] = Field(max_length=256)
    odd_component_count: StrictInt = Field(ge=0, le=256)
    upper_bound: StrictInt = Field(ge=0, le=128)

    @model_validator(mode="after")
    def require_canonical_barrier(self) -> Self:
        if tuple(sorted(self.barrier_vertices)) != self.barrier_vertices or len(
            set(self.barrier_vertices)
        ) != len(self.barrier_vertices):
            raise PydanticCustomError(
                "graph.tutte_berge_barrier_vertices_must_be_unique_and_",
                "Tutte-Berge barrier vertices must be unique and sorted",
            )
        return self


class GraphMaximumMatchingResult(StrictModel):
    maximum_matching_cardinality: StrictInt = Field(ge=0, le=128)
    witness_edges: tuple[tuple[GraphVertex, GraphVertex], ...]
    certificate: GraphTutteBergeCertificate

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if len(self.witness_edges) != self.maximum_matching_cardinality:
            raise PydanticCustomError(
                "graph.matching_witness_cardinality_must_match_the_resu",
                "matching witness cardinality must match the result",
            )
        if self.certificate.upper_bound != self.maximum_matching_cardinality:
            raise PydanticCustomError(
                "graph.tutte_berge_upper_bound_must_match_the_result",
                "Tutte-Berge upper bound must match the result",
            )
        if (
            any(left >= right for left, right in self.witness_edges)
            or tuple(sorted(self.witness_edges)) != self.witness_edges
            or len({vertex for edge in self.witness_edges for vertex in edge})
            != 2 * len(self.witness_edges)
        ):
            raise PydanticCustomError(
                "graph.matching_witness_must_be_canonical_and_vertex_di",
                "matching witness must be canonical and vertex-disjoint",
            )
        return self


class GraphTriangleCountResult(StrictModel):
    triangle_count: StrictInt = Field(ge=0, le=2_763_520)


class GraphCoreRequest(GraphInvariantRequest):
    k: StrictInt = Field(ge=0, le=255)


class GraphCoreResult(StrictModel):
    k: StrictInt = Field(ge=0, le=255)
    vertices: tuple[GraphVertex, ...]

    @model_validator(mode="after")
    def require_canonical_vertices(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices:
            raise PydanticCustomError(
                "graph.k_core_vertices_must_be_canonically_sorted",
                "k-core vertices must be canonically sorted",
            )
        if len(set(self.vertices)) != len(self.vertices):
            raise PydanticCustomError(
                "graph.k_core_vertices_must_be_unique", "k-core vertices must be unique"
            )
        return self


class GraphRadiusResult(StrictModel):
    status: Literal["COMPUTED", "NOT_APPLICABLE"]
    radius: StrictInt | None = Field(default=None, ge=0, le=255)
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
                raise PydanticCustomError(
                    "graph.computed_radius_requires_exact_value_connected",
                    "computed radius requires an exact value on a connected graph",
                )
        elif (
            self.radius is not None
            or self.connected
            or self.exactness != "NOT_APPLICABLE"
            or self.detail is None
        ):
            raise PydanticCustomError(
                "graph.inapplicable_radius_requires_no_value_explicit_detail",
                "inapplicable radius requires no value and an explicit detail",
            )
        return self


class GraphCardinalityMaximumResult(StrictModel):
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
            raise PydanticCustomError(
                "graph.witness_cardinality_must_match_the_incumbent",
                "witness cardinality must match the incumbent",
            )
        if tuple(sorted(self.witness_vertices)) != self.witness_vertices:
            raise PydanticCustomError(
                "graph.witness_vertices_must_be_canonically_sorted",
                "witness vertices must be canonically sorted",
            )
        if self.lower_bound != self.incumbent_value:
            raise PydanticCustomError(
                "graph.a_maximum_search_incumbent_is_the_lower_bound",
                "a maximum-search incumbent is the lower bound",
            )
        if self.status == "EXACT":
            if (
                self.optimum_value is None
                or self.lower_bound != self.optimum_value
                or self.upper_bound != self.optimum_value
            ):
                raise PydanticCustomError(
                    "graph.exact_result_must_bind_one_coincident_optimum",
                    "exact result must bind one coincident optimum",
                )
        elif self.optimum_value is not None:
            raise PydanticCustomError(
                "graph.incomplete_search_cannot_claim_an_optimum",
                "incomplete search cannot claim an optimum",
            )
        return self


class GraphCliqueNumberResult(GraphCardinalityMaximumResult):
    convention: Literal["MAXIMUM_COMPLETE_VERTEX_SUBSET"] = (
        "MAXIMUM_COMPLETE_VERTEX_SUBSET"
    )


class GraphCardinalityMaximumObligation(StrictModel):
    graph: ChromaticGraph
    predicate: Literal["GRAPH_CLIQUE_NUMBER_OPTIMALITY",]
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
