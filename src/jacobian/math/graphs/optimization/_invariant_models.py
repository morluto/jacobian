"""Typed contracts for finite simple-graph invariants."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._coloring_models import (
    ChromaticGraph,
    GraphVertex,
)
from jacobian.math.graphs.optimization._models import (
    OptimizationSearchStep,
    OptimizationStatus,
    OptimizationTermination,
)


class GraphInvariantRequest(StrictModel):
    graph: ChromaticGraph


class GraphMaximumMatchingGraph(ChromaticGraph):
    """A simple graph bounded for the polynomial-time matching operation."""

    vertices: tuple[GraphVertex, ...] = Field(max_length=64)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=2_016)


class GraphMaximumMatchingRequest(StrictModel):
    graph: GraphMaximumMatchingGraph


class GraphGirthResult(StrictModel):
    girth: StrictInt = Field(ge=0, le=32)
    has_cycle: StrictBool

    @model_validator(mode="after")
    def bind_cycle_status(self) -> Self:
        if self.has_cycle != (self.girth > 0):
            raise ValueError("has_cycle must agree with the girth sentinel")
        return self


class GraphDiameterResult(StrictModel):
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


class GraphEdgeConnectivityResult(StrictModel):
    edge_connectivity: StrictInt = Field(ge=0, le=31)


class GraphVertexConnectivityResult(StrictModel):
    vertex_connectivity: StrictInt = Field(ge=0, le=31)


class GraphEulerianResult(StrictModel):
    is_eulerian: StrictBool


class GraphSpanningTreeCountResult(StrictModel):
    spanning_tree_count: StrictInt = Field(ge=0)
    connected: StrictBool


class GraphTutteBergeCertificate(StrictModel):
    certificate_schema_version: Literal["1"] = "1"
    kind: Literal["TUTTE_BERGE_BARRIER"] = "TUTTE_BERGE_BARRIER"
    barrier_vertices: tuple[GraphVertex, ...] = Field(max_length=64)
    odd_component_count: StrictInt = Field(ge=0, le=64)
    upper_bound: StrictInt = Field(ge=0, le=32)

    @model_validator(mode="after")
    def require_canonical_barrier(self) -> Self:
        if tuple(sorted(self.barrier_vertices)) != self.barrier_vertices or len(
            set(self.barrier_vertices)
        ) != len(self.barrier_vertices):
            raise ValueError("Tutte-Berge barrier vertices must be unique and sorted")
        return self


class GraphMaximumMatchingResult(StrictModel):
    maximum_matching_cardinality: StrictInt = Field(ge=0, le=32)
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


class GraphTriangleCountResult(StrictModel):
    triangle_count: StrictInt = Field(ge=0, le=4_960)


class GraphCoreRequest(GraphInvariantRequest):
    k: StrictInt = Field(ge=0, le=32)


class GraphCoreResult(StrictModel):
    k: StrictInt = Field(ge=0, le=32)
    vertices: tuple[GraphVertex, ...]

    @model_validator(mode="after")
    def require_canonical_vertices(self) -> Self:
        if tuple(sorted(self.vertices)) != self.vertices:
            raise ValueError("k-core vertices must be canonically sorted")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("k-core vertices must be unique")
        return self


class GraphRadiusResult(StrictModel):
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


class GraphCardinalityMaximumObligation(StrictModel):
    obligation_schema_version: Literal["1"] = "1"
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
