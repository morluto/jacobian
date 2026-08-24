"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_COLORING_VERTICES = 64
MAX_EDGE_COLORING_EDGES = (
    MAX_EDGE_COLORING_VERTICES * (MAX_EDGE_COLORING_VERTICES - 1) // 2
)
# Polynomial-time checks (independent_set.maximal, edge_coloring.check) scale
# as O(V+E) and O(E^2) respectively and can use the full SimpleGraph limit
# of 64 vertices (SimpleUndirectedGraph max 256 is a future step). SAT-based
# operations (k_colorability.decide, edge_coloring.k_decide) use the same
# 64-vertex envelope but remain bounded by the explicit solver conflict
# budget (MAX_SOLVER_CONFLICT_BUDGET) per request.
DEFAULT_SOLVER_CONFLICT_BUDGET = 100_000
"""Default request-visible SAT conflict budget for one colorability decision."""

MAX_SOLVER_CONFLICT_BUDGET = 1_000_000
"""Upper bound on the accepted SAT conflict budget."""


def _run_edge_coloring_solver(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[str, tuple[int, ...] | None]:
    """Run one bounded exact edge-coloring SAT check.

    Returns ``("sat", coloring)``, ``("unsat", None)``, or
    ``("unknown", None)``.  Non-colorability is only ever claimed on an
    explicit ``unsat``; an exhausted budget reports ``unknown`` so the
    caller returns the typed incomplete outcome.
    """

    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    edge_colors = [z3.Int(f"c_{i}") for i in range(len(graph.edges))]
    solver.add(*(z3.And(c >= 0, c < colors) for c in edge_colors))
    for a, b in _incident_edge_index_pairs_for_canonical_graph(graph):
        solver.add(edge_colors[a] != edge_colors[b])
    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return "sat", tuple(model.eval(c).as_long() for c in edge_colors)
    if result == z3.unsat:
        return "unsat", None
    return "unknown", None


def _run_k_colorability_solver(
    graph: GraphEdgeList,
    colors: int,
    solver_conflicts: int,
) -> tuple[str, tuple[int, ...] | None]:
    """Run one bounded exact vertex-coloring SAT check.

    Returns ``("sat", coloring)``, ``("unsat", None)``, or
    ``("unknown", None)``; the coloring assigns one color in
    ``0..colors-1`` to each vertex index ``0..vertex_count-1``.
    Non-colorability is only ever claimed on an explicit ``unsat``; an
    exhausted budget reports ``unknown`` so the caller returns the typed
    incomplete outcome.
    """

    import z3

    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    vertex_colors = [z3.Int(f"color_{vertex}") for vertex in range(graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < colors) for color in vertex_colors))
    solver.add(*(vertex_colors[u] != vertex_colors[v] for u, v in graph.edges))
    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return "sat", tuple(model.eval(color).as_long() for color in vertex_colors)
    if result == z3.unsat:
        return "unsat", None
    return "unknown", None


def _is_proper_vertex_coloring(
    graph: GraphEdgeList,
    coloring: tuple[int, ...],
) -> bool:
    """Check whether a coloring assigns distinct colors to adjacent vertices."""

    return all(coloring[u] != coloring[v] for u, v in graph.edges)


def _edge_coloring_graph_schema() -> JsonSchemaValue:
    """Project the edge-coloring input bounds onto the shared graph schema."""

    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A simple undirected graph accepted by the edge-coloring operations: "
        f"at most {MAX_EDGE_COLORING_VERTICES} vertices and at most "
        f"{MAX_EDGE_COLORING_EDGES} edges."
    )
    schema["properties"]["vertices"].update(maxItems=MAX_EDGE_COLORING_VERTICES)
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


def _is_proper_edge_coloring(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
) -> bool:
    """Check whether a coloring assigns distinct colors to incident edges."""
    for a, b in _incident_edge_index_pairs_for_canonical_graph(graph):
        if coloring[a] == coloring[b]:
            return False
    return True


def _require_edge_coloring_graph_bound(graph: SimpleUndirectedGraph) -> None:
    if len(graph.vertices) > MAX_EDGE_COLORING_VERTICES:
        raise ValueError(
            f"edge-coloring supports at most {MAX_EDGE_COLORING_VERTICES} vertices"
        )


def _require_coloring_sequence(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
    colors: int,
) -> None:
    if len(coloring) != len(graph.edges):
        raise ValueError("coloring must assign one color per edge")
    for value in coloring:
        if not 0 <= value < colors:
            raise ValueError("coloring values must be in 0..colors-1")


def _require_conflicting_pair(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
    blocking_edge: tuple[str, str],
    conflicting_edge: tuple[str, str],
) -> None:
    if blocking_edge == conflicting_edge:
        raise ValueError("conflicting edge pair must be distinct")
    edge_index = {edge: idx for idx, edge in enumerate(graph.edges)}
    for edge in (blocking_edge, conflicting_edge):
        if edge[0] >= edge[1]:
            raise ValueError("blocking edges must be canonical pairs with left < right")
        if edge not in edge_index:
            raise ValueError("blocking edges must be edges of the graph")
    if not set(blocking_edge) & set(conflicting_edge):
        raise ValueError("conflicting edges must share a vertex")
    if coloring[edge_index[blocking_edge]] != coloring[edge_index[conflicting_edge]]:
        raise ValueError("conflicting edges must have the same color")


class GraphEdgeList(StrictModel):
    """A simple undirected graph given by an edge list."""

    # SAT instances are bounded by the explicit solver conflict budget but
    # capped at 64 vertices for one direct stateless call; polynomial-time
    # checks (maximal independent set) use the same envelope and run in
    # O(V+E).
    vertex_count: int = Field(ge=1, le=64)
    edges: tuple[tuple[int, int], ...] = Field(
        max_length=MAX_EDGE_COLORING_EDGES,
    )

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("a simple graph cannot contain self-loops")
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise ValueError("a simple graph cannot contain duplicate edges")
            seen.add(edge)
        return self


class KColorabilityRequest(StrictModel):
    """Decide whether a bounded simple graph admits a proper ``k``-coloring."""

    graph: GraphEdgeList
    colors: int = Field(ge=1, le=64)
    solver_conflicts: int = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
        description=(
            "Request-visible SAT work budget: the exact solver is cut off "
            "after this many conflict clauses; an exhausted budget yields "
            "the typed SOLVER_BUDGET_EXCEEDED outcome instead of an "
            "unbounded wait or a negative conclusion."
        ),
    )


class KColorabilityResult(StrictModel):
    """Whether a proper ``k``-coloring exists, with one coloring witness."""

    graph: GraphEdgeList
    colors: int = Field(ge=1, le=64)
    solver_conflicts: int = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
    )
    status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED"] = "DECIDED"
    colorable: bool | None = None
    coloring: tuple[int, ...] | None = None
    vertex_count: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def require_claim_consistency(self) -> Self:
        if self.vertex_count != self.graph.vertex_count:
            raise ValueError("vertex_count must equal the graph's vertex count")
        if self.status == "SOLVER_BUDGET_EXCEEDED":
            _require_k_colorability_budget_exceeded_shape(self)
            return self
        if self.colorable is None:
            raise ValueError("a decided result must claim colorable true or false")
        if self.colorable:
            _require_k_colorability_positive_witness(self)
        else:
            _require_k_colorability_negative_replay(self)
        return self


def _require_k_colorability_budget_exceeded_shape(
    result: KColorabilityResult,
) -> None:
    """A budget-exceeded outcome carries no claim and must replay unknown."""

    if result.colorable is not None or result.coloring is not None:
        raise ValueError("a budget-exceeded outcome carries no colorability claim")
    if not result.graph.edges:
        raise ValueError("empty graph is decided colorable without any search")
    if (
        _run_k_colorability_solver(
            result.graph, result.colors, result.solver_conflicts
        )[0]
        != "unknown"
    ):
        raise ValueError(
            "claimed solver-budget exceedance is not reproduced by the bounded replay"
        )


def _require_k_colorability_positive_witness(result: KColorabilityResult) -> None:
    """A colorable claim must carry a proper source-bound witness."""

    if result.coloring is None:
        raise ValueError("a colorable result must carry a coloring witness")
    if len(result.coloring) != result.graph.vertex_count:
        raise ValueError("coloring must assign one color per vertex")
    if any(not 0 <= color < result.colors for color in result.coloring):
        raise ValueError("coloring values must be in 0..colors-1")
    if not _is_proper_vertex_coloring(result.graph, result.coloring):
        raise ValueError("coloring witness must be a proper vertex coloring")


def _require_k_colorability_negative_replay(result: KColorabilityResult) -> None:
    """Replay non-colorability; only an explicit unsat may support it."""

    if result.coloring is not None:
        raise ValueError("a non-colorable result must not carry a coloring")
    if not result.graph.edges:
        raise ValueError("empty graph is k-colorable but result claims not colorable")
    if (
        _run_k_colorability_solver(
            result.graph, result.colors, result.solver_conflicts
        )[0]
        != "unsat"
    ):
        raise ValueError(
            "graph is k-colorable or undecided but result claims not colorable"
        )


class MaximalIndependentSetRequest(StrictModel):
    """One canonical candidate set in a bounded simple graph."""

    graph: GraphEdgeList
    candidate_set: tuple[int, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def require_canonical_candidate_set(self) -> Self:
        if tuple(sorted(self.candidate_set)) != self.candidate_set:
            raise ValueError("candidate_set must be strictly increasing")
        if len(set(self.candidate_set)) != len(self.candidate_set):
            raise ValueError("candidate_set must not contain duplicate vertices")
        if any(
            vertex < 0 or vertex >= self.graph.vertex_count
            for vertex in self.candidate_set
        ):
            raise ValueError("candidate vertices must lie in 0..vertex_count-1")
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
                raise ValueError("a maximal result must not carry a rejection witness")
            return self
        if self.decision == "NOT_INDEPENDENT":
            if self.blocking_edge is None or self.addable_vertex is not None:
                raise ValueError(
                    "a non-independent result requires exactly one blocking edge"
                )
            u, v = self.blocking_edge
            if u < 0 or v < 0 or u >= v:
                raise ValueError("blocking_edge must be a canonical pair u < v")
            return self
        if self.blocking_edge is not None or self.addable_vertex is None:
            raise ValueError(
                "an independent non-maximal result requires exactly one addable vertex"
            )
        if self.addable_vertex < 0:
            raise ValueError("addable_vertex must be nonnegative")
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
    colors: StrictInt = Field(ge=1, le=64)
    coloring: tuple[StrictInt, ...] = Field(
        max_length=MAX_EDGE_COLORING_EDGES,
        description=(
            "Edge colors aligned to graph.edges: coloring[i] is the color of "
            "graph.edges[i] in 0..colors-1 (graph.edges order is authoritative)."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_assignment(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        _require_coloring_sequence(self.graph, self.coloring, self.colors)
        return self


def _decided_unsat_result(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> EdgeKColorabilityResult:
    """Build a decided-negative result from one explicit bounded unsat.

    Direct construction from the producing solve skips result replay so one
    declared budget covers all solver work; independently supplied results
    always validate through ``_require_negative_replay``.
    """

    return EdgeKColorabilityResult.model_construct(
        graph=graph,
        colors=colors,
        solver_conflicts=solver_conflicts,
        status="DECIDED",
        colorable=False,
        coloring=None,
        edge_count=len(graph.edges),
    )


def _budget_exceeded_result(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> EdgeKColorabilityResult:
    """Build the typed incomplete outcome from one explicit bounded unknown.

    As with ``_decided_unsat_result``, the producing solve's own answer is
    carried unclaimed; replay stays reserved for independently supplied
    results via ``_require_budget_exceeded_shape``.
    """

    return EdgeKColorabilityResult.model_construct(
        graph=graph,
        colors=colors,
        solver_conflicts=solver_conflicts,
        status="SOLVER_BUDGET_EXCEEDED",
        colorable=None,
        coloring=None,
        edge_count=len(graph.edges),
    )


def _require_budget_exceeded_shape(result: EdgeKColorabilityResult) -> None:
    """A budget-exceeded outcome carries no claim and must replay unknown."""

    if result.colorable is not None or result.coloring is not None:
        raise ValueError("a budget-exceeded outcome carries no colorability claim")
    if not result.graph.edges:
        raise ValueError("empty graph is decided colorable without any search")
    if (
        _run_edge_coloring_solver(result.graph, result.colors, result.solver_conflicts)[
            0
        ]
        != "unknown"
    ):
        raise ValueError(
            "claimed solver-budget exceedance is not reproduced by the bounded replay"
        )


def _require_negative_replay(result: EdgeKColorabilityResult) -> None:
    """Replay non-colorability; only an explicit unsat may support it."""

    if result.coloring is not None:
        raise ValueError("a non-colorable result must not carry a coloring")
    if not result.graph.edges:
        raise ValueError(
            "empty graph is k-edge-colorable but result claims not colorable"
        )
    if (
        _run_edge_coloring_solver(result.graph, result.colors, result.solver_conflicts)[
            0
        ]
        != "unsat"
    ):
        raise ValueError(
            "graph is k-edge-colorable or undecided but result claims not colorable"
        )


def _require_positive_witness(result: EdgeKColorabilityResult) -> None:
    """A colorable claim must carry a proper source-bound witness."""

    if result.coloring is None:
        raise ValueError("a colorable result must carry a coloring witness")
    if result.coloring.graph != result.graph or result.coloring.colors != result.colors:
        raise ValueError("witness must bind the result's own graph and palette")
    if not _is_proper_edge_coloring(result.graph, result.coloring.coloring):
        raise ValueError("coloring witness must be a proper edge coloring")


class EdgeKColorabilityRequest(StrictModel):
    """Decide whether a simple graph admits a proper ``k``-edge-coloring."""

    graph: EdgeColoringGraph
    colors: StrictInt = Field(ge=1, le=64)
    solver_conflicts: StrictInt = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
        description=(
            "Request-visible SAT work budget: the exact solver is cut off "
            "after this many conflict clauses; an exhausted budget yields "
            "the typed SOLVER_BUDGET_EXCEEDED outcome instead of an "
            "unbounded wait or a negative conclusion."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_graph(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        return self


class EdgeKColorabilityResult(StrictModel):
    """Whether a proper ``k``-edge-coloring exists, with one coloring witness."""

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=64)
    solver_conflicts: StrictInt = Field(
        default=DEFAULT_SOLVER_CONFLICT_BUDGET,
        ge=1,
        le=MAX_SOLVER_CONFLICT_BUDGET,
    )
    status: Literal["DECIDED", "SOLVER_BUDGET_EXCEEDED"] = "DECIDED"
    colorable: bool | None = None
    coloring: EdgeColoringAssignment | None = None
    edge_count: StrictInt = Field(ge=0, le=MAX_EDGE_COLORING_EDGES)

    @model_validator(mode="after")
    def require_witness_consistency(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        if self.edge_count != len(self.graph.edges):
            raise ValueError("edge_count must equal the number of graph edges")
        if self.status == "SOLVER_BUDGET_EXCEEDED":
            _require_budget_exceeded_shape(self)
            return self
        if self.colorable is None:
            raise ValueError("a decided result must claim colorable true or false")
        if self.colorable:
            _require_positive_witness(self)
        else:
            _require_negative_replay(self)
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

    @model_validator(mode="after")
    def require_blocking_edge_consistency(self) -> Self:
        actual_proper = _is_proper_edge_coloring(
            self.assignment.graph, self.assignment.coloring
        )
        if self.proper != actual_proper:
            raise ValueError("proper flag does not match the submitted coloring")
        if self.proper:
            if self.blocking_edge is not None or self.conflicting_edge is not None:
                raise ValueError("a proper coloring must not carry a blocking edge")
            return self
        if self.blocking_edge is None or self.conflicting_edge is None:
            raise ValueError("an improper coloring must carry a conflicting edge pair")
        _require_conflicting_pair(
            self.assignment.graph,
            self.assignment.coloring,
            self.blocking_edge,
            self.conflicting_edge,
        )
        return self
