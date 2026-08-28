"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)

# The old 64-vertex envelope admitted a complete 64-vertex graph, hence at
# most C(64, 2) edge variables.  Retain that formula/output envelope rather
# than treating 64 vertices as the mathematical domain: sparse graphs on the
# shared 256-vertex axis are no more expensive in the vertex-coloring formula.
MAX_COLORING_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
_DENSE_COLORING_REFERENCE_ORDER = 64
MAX_COLORING_COLORS = _DENSE_COLORING_REFERENCE_ORDER
MAX_EDGE_COLORING_EDGES = (
    _DENSE_COLORING_REFERENCE_ORDER * (_DENSE_COLORING_REFERENCE_ORDER - 1) // 2
)
# Edge coloring has one inequality for every incident pair of edges.  A K_64
# has 64 * C(63, 2) such pairs; this is the retained bound on the materialized
# edge-coloring formula and checker, independently of the vertex count.
MAX_EDGE_COLORING_CONFLICT_PAIRS = (
    _DENSE_COLORING_REFERENCE_ORDER
    * (_DENSE_COLORING_REFERENCE_ORDER - 1)
    * (_DENSE_COLORING_REFERENCE_ORDER - 2)
    // 2
)
# Polynomial-time checks (independent_set.maximal, edge_coloring.check) and
# SAT-based operations use the shared 256-vertex graph axis.  The derived
# edge-variable and incident-pair limits above bound formula construction;
# SAT search itself remains bounded by the explicit conflict budget.
DEFAULT_SOLVER_CONFLICT_BUDGET = 100_000
"""Default request-visible SAT conflict budget for one colorability decision."""

MAX_SOLVER_CONFLICT_BUDGET = 1_000_000
"""Upper bound on the accepted SAT conflict budget."""


def _require_indexed_coloring_graph(graph: IndexedSimpleUndirectedGraph) -> None:
    if len(graph.edges) > MAX_EDGE_COLORING_EDGES:
        raise PydanticCustomError(
            "graph.coloring_edge_count_exceeds_formula_bound",
            "coloring operations support at most "
            f"{MAX_EDGE_COLORING_EDGES} adjacency constraints",
        )


def _indexed_coloring_graph_schema() -> JsonSchemaValue:
    """Project the coloring envelope onto the shared indexed graph value."""

    schema = IndexedSimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "An integer-indexed simple undirected graph accepted by the coloring "
        f"operations: at most {MAX_COLORING_VERTICES} vertices and at most "
        f"{MAX_EDGE_COLORING_EDGES} adjacency constraints. Sparse graphs may "
        "use the full vertex axis."
    )
    schema["properties"]["vertex_count"].update(maximum=MAX_COLORING_VERTICES)
    schema["properties"]["edges"].update(maxItems=MAX_EDGE_COLORING_EDGES)
    return schema


IndexedColoringGraph = Annotated[
    IndexedSimpleUndirectedGraph,
    WithJsonSchema(_indexed_coloring_graph_schema()),
]


def _edge_coloring_graph_schema() -> JsonSchemaValue:
    """Project the edge-coloring input bounds onto the shared graph schema."""

    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A simple undirected graph accepted by the edge-coloring operations: "
        f"at most {MAX_COLORING_VERTICES} vertices, at most "
        f"{MAX_EDGE_COLORING_EDGES} edges, and at most "
        f"{MAX_EDGE_COLORING_CONFLICT_PAIRS} incident-edge constraints."
    )
    schema["properties"]["vertices"].update(maxItems=MAX_COLORING_VERTICES)
    schema["properties"]["edges"].update(maxItems=MAX_EDGE_COLORING_EDGES)
    return schema


EdgeColoringGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_edge_coloring_graph_schema()),
]


def _incident_edge_index_pairs_for_canonical_graph(
    graph: SimpleUndirectedGraph,
) -> list[tuple[int, int]]:
    """Return pairs of edge indices that share a vertex (must differ in color)."""
    incidence: dict[str, list[int]] = {}
    for edge_index, (u, v) in enumerate(graph.edges):
        incidence.setdefault(u, []).append(edge_index)
        incidence.setdefault(v, []).append(edge_index)
    pairs: list[tuple[int, int]] = []
    for indices in incidence.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    return pairs


def _edge_coloring_conflict_pair_count(graph: SimpleUndirectedGraph) -> int:
    """Count edge-pair inequalities without materializing the pair list."""

    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    return sum(degree * (degree - 1) // 2 for degree in degrees.values())


def _require_edge_coloring_graph_bound(graph: SimpleUndirectedGraph) -> None:
    if len(graph.edges) > MAX_EDGE_COLORING_EDGES:
        raise PydanticCustomError(
            "graph.edge_coloring_edge_count_exceeds_formula_bound",
            f"edge-coloring supports at most {MAX_EDGE_COLORING_EDGES} edges",
        )
    if _edge_coloring_conflict_pair_count(graph) > MAX_EDGE_COLORING_CONFLICT_PAIRS:
        raise PydanticCustomError(
            "graph.edge_coloring_incident_pair_count_exceeds_formula_bound",
            "edge-coloring supports at most "
            f"{MAX_EDGE_COLORING_CONFLICT_PAIRS} incident-edge constraints",
        )


def _require_coloring_sequence(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
    colors: int,
) -> None:
    if len(coloring) != len(graph.edges):
        raise PydanticCustomError(
            "graph.coloring_must_assign_one_color_per_edge",
            "coloring must assign one color per edge",
        )
    for value in coloring:
        if not 0 <= value < colors:
            raise PydanticCustomError(
                "graph.coloring_values_must_be_in_0_colors_1",
                "coloring values must be in 0..colors-1",
            )


class KColorabilityRequest(StrictModel):
    """Decide whether a bounded simple graph admits a proper ``k``-coloring."""

    graph: IndexedColoringGraph
    colors: int = Field(ge=1, le=MAX_COLORING_COLORS)
    solver_conflicts: int = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
        description=(
            "Request-visible SAT work budget: the exact solver is cut off "
            "after this many conflict clauses; that exhaustion yields "
            "SOLVER_BUDGET_EXCEEDED, while execution failures are reported "
            "separately without a mathematical conclusion."
        ),
    )


class KColorabilityResult(StrictModel):
    """Whether a proper ``k``-coloring exists, with one coloring witness."""

    graph: IndexedColoringGraph
    colors: int = Field(ge=1, le=MAX_COLORING_COLORS)
    solver_conflicts: int = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
    )
    status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"] = "DECIDED"
    colorable: bool | None = None
    coloring: tuple[int, ...] | None = None
    vertex_count: int = Field(ge=0, le=MAX_COLORING_VERTICES)

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: IndexedSimpleUndirectedGraph,
        colors: int,
        solver_conflicts: int,
        status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"],
        colorable: bool | None,
        coloring: tuple[int, ...] | None,
    ) -> Self:
        """Construct a result already established by the owner-local kernel."""

        return cls.model_construct(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status=status,
            colorable=colorable,
            coloring=coloring,
            vertex_count=graph.vertex_count,
        )

    @model_validator(mode="after")
    def require_claim_consistency(self) -> Self:
        if self.vertex_count != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.vertex_count_must_equal_the_graph_s_vertex_count",
                "vertex_count must equal the graph's vertex count",
            )
        if self.status in {"SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"}:
            _require_k_colorability_budget_exceeded_shape(self)
            return self
        if self.colorable is None:
            raise PydanticCustomError(
                "graph.a_decided_result_must_claim_colorable_true_or_fa",
                "a decided result must claim colorable true or false",
            )
        if self.colorable:
            _require_k_colorability_positive_witness(self)
        else:
            _require_k_colorability_negative_shape(self)
        return self


def _require_k_colorability_budget_exceeded_shape(
    result: KColorabilityResult,
) -> None:
    """A budget-exceeded outcome carries no mathematical claim."""

    if result.colorable is not None or result.coloring is not None:
        raise PydanticCustomError(
            "graph.a_budget_exceeded_outcome_carries_no_colorabilit",
            "a budget-exceeded outcome carries no colorability claim",
        )
    if not result.graph.edges:
        raise PydanticCustomError(
            "graph.empty_graph_is_decided_colorable_without_any_sea",
            "empty graph is decided colorable without any search",
        )


def _require_k_colorability_positive_witness(result: KColorabilityResult) -> None:
    """A colorable claim must carry a proper source-bound witness."""

    if result.coloring is None:
        raise PydanticCustomError(
            "graph.a_colorable_result_must_carry_a_coloring_witness",
            "a colorable result must carry a coloring witness",
        )
    if len(result.coloring) != result.graph.vertex_count:
        raise PydanticCustomError(
            "graph.coloring_must_assign_one_color_per_vertex",
            "coloring must assign one color per vertex",
        )
    if any(not 0 <= color < result.colors for color in result.coloring):
        raise PydanticCustomError(
            "graph.coloring_values_must_be_in_0_colors_1",
            "coloring values must be in 0..colors-1",
        )


def _require_k_colorability_negative_shape(result: KColorabilityResult) -> None:
    """A non-colorability claim carries no positive witness."""

    if result.coloring is not None:
        raise PydanticCustomError(
            "graph.a_non_colorable_result_must_not_carry_a_coloring",
            "a non-colorable result must not carry a coloring",
        )
    if not result.graph.edges:
        raise PydanticCustomError(
            "graph.empty_graph_is_k_colorable_but_result_claims_not",
            "empty graph is k-colorable but result claims not colorable",
        )


class MaximalIndependentSetRequest(StrictModel):
    """One canonical candidate set in a bounded simple graph."""

    graph: IndexedColoringGraph
    candidate_set: tuple[int, ...] = Field(max_length=MAX_COLORING_VERTICES)

    @model_validator(mode="after")
    def require_canonical_candidate_set(self) -> Self:
        if tuple(sorted(self.candidate_set)) != self.candidate_set:
            raise PydanticCustomError(
                "graph.candidate_set_must_be_strictly_increasing",
                "candidate_set must be strictly increasing",
            )
        if len(set(self.candidate_set)) != len(self.candidate_set):
            raise PydanticCustomError(
                "graph.candidate_set_must_not_contain_duplicate_vertice",
                "candidate_set must not contain duplicate vertices",
            )
        if any(
            vertex < 0 or vertex >= self.graph.vertex_count
            for vertex in self.candidate_set
        ):
            raise PydanticCustomError(
                "graph.candidate_vertices_must_lie_in_0_vertex_count_1",
                "candidate vertices must lie in 0..vertex_count-1",
            )
        return self


class MaximalIndependentSetResult(StrictModel):
    """A closed decision with a concrete rejection witness when applicable."""

    decision: Literal["MAXIMAL", "NOT_INDEPENDENT", "INDEPENDENT_NOT_MAXIMAL"]
    blocking_edge: tuple[int, int] | None = None
    addable_vertex: int | None = None

    @model_validator(mode="after")
    def bind_witness_to_decision(self) -> Self:
        if self.decision == "MAXIMAL":
            if self.blocking_edge is not None or self.addable_vertex is not None:
                raise PydanticCustomError(
                    "graph.a_maximal_result_must_not_carry_a_rejection_witn",
                    "a maximal result must not carry a rejection witness",
                )
            return self
        if self.decision == "NOT_INDEPENDENT":
            if self.blocking_edge is None or self.addable_vertex is not None:
                raise PydanticCustomError(
                    "graph.non_independent_result_requires_exactly_one_blocking",
                    "a non-independent result requires exactly one blocking edge",
                )
            u, v = self.blocking_edge
            if u < 0 or v < 0 or u >= v:
                raise PydanticCustomError(
                    "graph.blocking_edge_must_be_a_canonical_pair_u_v",
                    "blocking_edge must be a canonical pair u < v",
                )
            return self
        if self.blocking_edge is not None or self.addable_vertex is None:
            raise PydanticCustomError(
                "graph.independent_non_maximal_result_requires_exactly_one",
                "an independent non-maximal result requires exactly one addable vertex",
            )
        if self.addable_vertex < 0:
            raise PydanticCustomError(
                "graph.addable_vertex_must_be_nonnegative",
                "addable_vertex must be nonnegative",
            )
        return self


# ---------------------------------------------------------------------------
# Edge coloring
# ---------------------------------------------------------------------------


class EdgeColoringAssignment(StrictModel):
    """Canonical source-bound edge-coloring value.

    One simple undirected graph, one palette size, and one color per edge:
    ``coloring[i]`` is the color of ``graph.edges[i]`` in ``0..colors-1``
    (graph.edges order is authoritative).  Returned by
    ``graph.edge_coloring.k_decide`` for colorable graphs and accepted
    unchanged by ``graph.edge_coloring.check``, which decides properness;
    structural validity only, so improper candidates are representable.
    """

    graph: EdgeColoringGraph
    colors: StrictInt = Field(ge=1, le=MAX_COLORING_COLORS)
    coloring: tuple[StrictInt, ...] = Field(
        max_length=MAX_EDGE_COLORING_EDGES,
        description=(
            "Edge colors aligned to graph.edges: coloring[i] is the color of "
            "graph.edges[i] in 0..colors-1 (graph.edges order is authoritative)."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_assignment(self) -> Self:
        _require_coloring_sequence(self.graph, self.coloring, self.colors)
        return self


def _require_budget_exceeded_shape(result: EdgeKColorabilityResult) -> None:
    """A budget-exceeded outcome carries no mathematical claim."""

    if result.colorable is not None or result.coloring is not None:
        raise PydanticCustomError(
            "graph.a_budget_exceeded_outcome_carries_no_colorabilit",
            "a budget-exceeded outcome carries no colorability claim",
        )
    if not result.graph.edges:
        raise PydanticCustomError(
            "graph.empty_graph_is_decided_colorable_without_any_sea",
            "empty graph is decided colorable without any search",
        )


def _require_negative_shape(result: EdgeKColorabilityResult) -> None:
    """A non-colorability claim carries no positive witness."""

    if result.coloring is not None:
        raise PydanticCustomError(
            "graph.a_non_colorable_result_must_not_carry_a_coloring",
            "a non-colorable result must not carry a coloring",
        )
    if not result.graph.edges:
        raise PydanticCustomError(
            "graph.empty_edge_colorable_but_result_claims_colorable",
            "empty graph is k-edge-colorable but result claims not colorable",
        )


def _require_positive_witness(result: EdgeKColorabilityResult) -> None:
    """A colorable claim must carry a proper source-bound witness."""

    if result.coloring is None:
        raise PydanticCustomError(
            "graph.a_colorable_result_must_carry_a_coloring_witness",
            "a colorable result must carry a coloring witness",
        )
    if result.coloring.graph != result.graph or result.coloring.colors != result.colors:
        raise PydanticCustomError(
            "graph.witness_must_bind_the_result_s_own_graph_and_pal",
            "witness must bind the result's own graph and palette",
        )


class EdgeKColorabilityRequest(StrictModel):
    """Decide whether a simple graph admits a proper ``k``-edge-coloring."""

    graph: EdgeColoringGraph
    colors: StrictInt = Field(ge=1, le=MAX_COLORING_COLORS)
    solver_conflicts: StrictInt = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
        description=(
            "Request-visible SAT work budget: the exact solver is cut off "
            "after this many conflict clauses; that exhaustion yields "
            "SOLVER_BUDGET_EXCEEDED, while execution failures are reported "
            "separately without a mathematical conclusion."
        ),
    )


class EdgeKColorabilityResult(StrictModel):
    """Whether a proper ``k``-edge-coloring exists, with one coloring witness."""

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=MAX_COLORING_COLORS)
    solver_conflicts: StrictInt = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
    )
    status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"] = "DECIDED"
    colorable: bool | None = None
    coloring: EdgeColoringAssignment | None = None
    edge_count: StrictInt = Field(ge=0, le=MAX_EDGE_COLORING_EDGES)

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        colors: int,
        solver_conflicts: int,
        status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"],
        colorable: bool | None,
        coloring: EdgeColoringAssignment | None,
    ) -> Self:
        """Construct a result already established by the owner-local kernel."""

        return cls.model_construct(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status=status,
            colorable=colorable,
            coloring=coloring,
            edge_count=len(graph.edges),
        )

    @model_validator(mode="after")
    def require_witness_consistency(self) -> Self:
        if self.edge_count != len(self.graph.edges):
            raise PydanticCustomError(
                "graph.edge_count_must_equal_the_number_of_graph_edges",
                "edge_count must equal the number of graph edges",
            )
        if self.status in {"SOLVER_BUDGET_EXCEEDED", "EXECUTION_FAILED"}:
            _require_budget_exceeded_shape(self)
            return self
        if self.colorable is None:
            raise PydanticCustomError(
                "graph.a_decided_result_must_claim_colorable_true_or_fa",
                "a decided result must claim colorable true or false",
            )
        if self.colorable:
            _require_positive_witness(self)
        else:
            _require_negative_shape(self)
        return self


class EdgeColoringCheckRequest(StrictModel):
    """Validate one source-bound edge-to-color assignment as a proper coloring."""

    assignment: EdgeColoringAssignment


class EdgeColoringCheckResult(StrictModel):
    """Whether the submitted edge coloring is proper, with a replayable conflict pair."""

    assignment: EdgeColoringAssignment
    proper: bool
    blocking_edge: tuple[str, str] | None = None
    conflicting_edge: tuple[str, str] | None = None

    @classmethod
    def _from_kernel(
        cls,
        *,
        assignment: EdgeColoringAssignment,
        proper: bool,
        blocking_edge: tuple[str, str] | None,
        conflicting_edge: tuple[str, str] | None,
    ) -> Self:
        return cls.model_construct(
            assignment=assignment,
            proper=proper,
            blocking_edge=blocking_edge,
            conflicting_edge=conflicting_edge,
        )

    @model_validator(mode="after")
    def require_blocking_edge_consistency(self) -> Self:
        if self.proper:
            if self.blocking_edge is not None or self.conflicting_edge is not None:
                raise PydanticCustomError(
                    "graph.a_proper_coloring_must_not_carry_a_blocking_edge",
                    "a proper coloring must not carry a blocking edge",
                )
            return self
        if self.blocking_edge is None or self.conflicting_edge is None:
            raise PydanticCustomError(
                "graph.an_improper_coloring_must_carry_a_conflicting_ed",
                "an improper coloring must carry a conflicting edge pair",
            )
        return self
