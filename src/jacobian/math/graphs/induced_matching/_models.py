"""Typed contracts for maximum induced matching."""

from pydantic import Field, StrictInt, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceSearchStatus,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_INDUCED_MATCHING_EDGES = 128
MAX_INDUCED_MATCHING_CONFLICT_PAIRS = (
    MAX_INDUCED_MATCHING_EDGES * (MAX_INDUCED_MATCHING_EDGES - 1) // 2
)
MAX_INDUCED_MATCHING_CONFLICT_GRAPH_CELLS = (
    MAX_INDUCED_MATCHING_EDGES + MAX_INDUCED_MATCHING_CONFLICT_PAIRS
)


class MaximumInducedMatchingRequest(StrictModel):
    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical finite simple graph. Its source edge count must not exceed "
            "resource_budget.max_order because one conflict vertex is created per edge."
        )
    )
    resource_budget: IndependenceNumberBudget = Field(
        default_factory=IndependenceNumberBudget,
        description=(
            "Bounded exact-independence search envelope for the private conflict "
            "graph; max_order must cover the source edge count."
        ),
    )


class SourceEdgeBinding(StrictModel):
    edge_id: StrictStr = Field(pattern=r"^e(?:0|[1-9][0-9]*)$")
    endpoints: tuple[StrictStr, StrictStr] = Field(
        description="Canonical source edge endpoints in lexicographic order."
    )

    @model_validator(mode="after")
    def require_canonical_endpoints(self) -> "SourceEdgeBinding":
        if self.endpoints[0] >= self.endpoints[1]:
            raise PydanticCustomError(
                "graph.induced_matching.edge_endpoints_must_be_canonical",
                "source edge endpoints must be distinct and lexicographically ordered",
            )
        return self


class MaximumInducedMatchingResult(StrictModel):
    graph: SimpleUndirectedGraph
    source_edges: tuple[SourceEdgeBinding, ...] = Field(
        max_length=MAX_INDUCED_MATCHING_EDGES
    )
    status: IndependenceSearchStatus
    cardinality: StrictInt = Field(ge=0, le=MAX_INDUCED_MATCHING_EDGES)
    lower_bound: StrictInt = Field(ge=0, le=MAX_INDUCED_MATCHING_EDGES)
    upper_bound: StrictInt = Field(ge=0, le=MAX_INDUCED_MATCHING_EDGES)
    selected_edge_ids: tuple[StrictStr, ...] = Field(
        max_length=MAX_INDUCED_MATCHING_EDGES
    )
    induced_endpoint_graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def bind_source_and_witness(self) -> "MaximumInducedMatchingResult":
        source_edges = tuple(sorted(self.graph.edges))
        bindings = tuple(binding.endpoints for binding in self.source_edges)
        expected_ids = tuple(f"e{index}" for index in range(len(source_edges)))
        actual_ids = tuple(binding.edge_id for binding in self.source_edges)
        if bindings != source_edges or actual_ids != expected_ids:
            raise PydanticCustomError(
                "graph.induced_matching.source_edge_axis_must_be_complete",
                "source_edges must contain every source edge in canonical order with contiguous IDs",
            )

        if self.selected_edge_ids != tuple(sorted(set(self.selected_edge_ids))):
            raise PydanticCustomError(
                "graph.induced_matching.selected_edge_ids_must_be_sorted_unique",
                "selected edge IDs must be unique and lexicographically sorted",
            )
        source_by_id = dict(zip(actual_ids, bindings, strict=True))
        if any(edge_id not in source_by_id for edge_id in self.selected_edge_ids):
            raise PydanticCustomError(
                "graph.induced_matching.selected_edge_must_belong_to_source",
                "every selected edge ID must belong to source_edges",
            )
        selected_edges = tuple(
            source_by_id[edge_id] for edge_id in self.selected_edge_ids
        )
        selected_vertices = {vertex for edge in selected_edges for vertex in edge}
        if len(selected_vertices) != 2 * len(selected_edges):
            raise PydanticCustomError(
                "graph.induced_matching.selected_edges_must_be_vertex_disjoint",
                "selected source edges must be pairwise vertex-disjoint",
            )
        expected_endpoint_edges = tuple(
            edge
            for edge in source_edges
            if edge[0] in selected_vertices and edge[1] in selected_vertices
        )
        if set(expected_endpoint_edges) != set(selected_edges):
            raise PydanticCustomError(
                "graph.induced_matching.selected_edges_must_induce_no_cross_edges",
                "selected source edges must be the complete source subgraph on their endpoints",
            )
        expected_endpoint_vertices = tuple(sorted(selected_vertices))
        if (
            self.induced_endpoint_graph.vertices != expected_endpoint_vertices
            or self.induced_endpoint_graph.edges != expected_endpoint_edges
        ):
            raise PydanticCustomError(
                "graph.induced_matching.endpoint_graph_must_be_complete",
                "induced_endpoint_graph must be the complete canonical source subgraph on selected endpoints",
            )
        if self.cardinality != len(self.selected_edge_ids):
            raise PydanticCustomError(
                "graph.induced_matching.cardinality_must_match_witness",
                "cardinality must equal the number of selected edge IDs",
            )
        if self.lower_bound != self.cardinality or not (
            self.lower_bound <= self.upper_bound <= len(source_edges)
        ):
            raise PydanticCustomError(
                "graph.induced_matching.bounds_must_contain_incumbent",
                "bounds must contain the returned feasible cardinality",
            )
        if self.status == "EXACT" and self.upper_bound != self.cardinality:
            raise PydanticCustomError(
                "graph.induced_matching.exact_bounds_must_coincide",
                "an exact induced matching result must have coincident bounds",
            )
        if self.status == "UNKNOWN" and self.upper_bound != len(source_edges):
            raise PydanticCustomError(
                "graph.induced_matching.unknown_upper_bound_must_be_source_edge_count",
                "an incomplete induced matching result must report the source edge count as its safe upper bound",
            )
        return self
