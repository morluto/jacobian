"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring._chromatic_number_models import (
    ChromaticNumberCertificateCheckResult,
    _evaluate_chromatic_number_certificate,
    _require_bounded_sources,
)
from jacobian.math.graphs.coloring._coloring_process import run_coloring_worker
from jacobian.math.graphs.coloring._models import (
    MAX_COLORING_COLORS,
    MAX_SOLVER_CONFLICT_BUDGET,
    EdgeColoringAssignment,
    EdgeColoringCheckResult,
    EdgeKColorabilityResult,
    KColorabilityResult,
    MaximalIndependentSetResult,
    _incident_edge_index_pairs_for_canonical_graph,
    _require_edge_coloring_graph_bound,
    _require_indexed_coloring_graph,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def _admit_chromatic_number_certificate(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> None:
    """Admit the exact replay and retained-result envelope."""
    try:
        _require_bounded_sources(
            graph,
            claimed_chromatic_number,
            coloring,
            weights,
        )
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error


def _admit_k_colorability(graph: IndexedSimpleUndirectedGraph) -> None:
    _admit_indexed_coloring_graph(graph)


def _admit_solver_parameters(colors: int, solver_conflicts: int) -> None:
    if not 1 <= colors <= MAX_COLORING_COLORS:
        raise OperationDomainValidationError(
            location=("colors",),
            code="graph.colors_must_be_between_1_and_maximum",
            message=f"colors must be in 1..{MAX_COLORING_COLORS}",
        )
    if not 1 <= solver_conflicts <= MAX_SOLVER_CONFLICT_BUDGET:
        raise OperationDomainValidationError(
            location=("solver_conflicts",),
            code="graph.solver_conflicts_must_be_between_1_and_maximum",
            message=(f"solver_conflicts must be in 1..{MAX_SOLVER_CONFLICT_BUDGET}"),
        )


def _admit_indexed_coloring_graph(graph: IndexedSimpleUndirectedGraph) -> None:
    try:
        _require_indexed_coloring_graph(graph)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("graph",), code=error.type, message=str(error)
        ) from error


def _admit_candidate_set(
    graph: IndexedSimpleUndirectedGraph, candidate_set: tuple[int, ...]
) -> None:
    if tuple(sorted(candidate_set)) != candidate_set:
        raise OperationDomainValidationError(
            location=("candidate_set",),
            code="graph.candidate_set_must_be_strictly_increasing",
            message="candidate_set must be strictly increasing",
        )
    if len(set(candidate_set)) != len(candidate_set):
        raise OperationDomainValidationError(
            location=("candidate_set",),
            code="graph.candidate_set_must_not_contain_duplicate_vertices",
            message="candidate_set must not contain duplicate vertices",
        )
    if any(not 0 <= vertex < graph.vertex_count for vertex in candidate_set):
        raise OperationDomainValidationError(
            location=("candidate_set",),
            code="graph.candidate_set_vertices_must_be_in_range",
            message="candidate_set vertices must be in 0..vertex_count-1",
        )


def _admit_edge_coloring_graph(graph: SimpleUndirectedGraph) -> None:
    try:
        _require_edge_coloring_graph_bound(graph)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("graph",), code=error.type, message=str(error)
        ) from error


def chromatic_number_certificate(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> ChromaticNumberCertificateCheckResult:
    """Check a proper coloring and exact fractional-clique lower certificate."""
    _admit_chromatic_number_certificate(
        graph, claimed_chromatic_number, coloring, weights
    )
    evaluation = _evaluate_chromatic_number_certificate(
        graph,
        claimed_chromatic_number,
        coloring,
        weights,
    )
    return ChromaticNumberCertificateCheckResult._from_kernel(
        graph=graph,
        claimed_chromatic_number=claimed_chromatic_number,
        coloring=coloring,
        weights=weights,
        evaluation=evaluation,
    )


def k_colorability(
    graph: IndexedSimpleUndirectedGraph, colors: int, solver_conflicts: int
) -> KColorabilityResult:
    """Decide whether a simple graph admits a proper ``k``-coloring.

    Uses a Z3 SAT search bounded by the caller-visible ``solver_conflicts``
    budget and returns one proper coloring as the witness of a colorable
    decision.  Non-colorability is claimed only on an explicit
    unsatisfiable outcome; an exhausted budget yields the typed
    ``SOLVER_BUDGET_EXCEEDED`` outcome instead of an unbounded wait.  The
    conflict budget bounds the SAT search after owner-local formula admission.
    """
    _admit_k_colorability(graph)
    _admit_solver_parameters(colors, solver_conflicts)
    if not graph.edges:
        return KColorabilityResult._from_kernel(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=(0,) * graph.vertex_count,
        )
    outcome, coloring = run_coloring_worker("vertex", graph, colors, solver_conflicts)
    if outcome == "sat":
        if coloring is None:
            raise AssertionError(
                "the bounded solver returned a satisfying outcome without a witness"
            )
        return KColorabilityResult._from_kernel(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=coloring,
        )
    if outcome == "unsat":
        return KColorabilityResult._from_kernel(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status="DECIDED",
            colorable=False,
            coloring=None,
        )
    return KColorabilityResult._from_kernel(
        graph=graph,
        colors=colors,
        solver_conflicts=solver_conflicts,
        status=(
            "SOLVER_BUDGET_EXCEEDED"
            if outcome == "budget_exceeded"
            else "EXECUTION_FAILED"
        ),
        colorable=None,
        coloring=None,
    )


def maximal_independent_set(
    graph: IndexedSimpleUndirectedGraph, candidate_set: tuple[int, ...]
) -> MaximalIndependentSetResult:
    """Decide maximal independence and return the first canonical obstruction."""
    _admit_indexed_coloring_graph(graph)
    _admit_candidate_set(graph, candidate_set)
    candidate = frozenset(candidate_set)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in graph.edges))
    for edge in edges:
        if edge[0] in candidate and edge[1] in candidate:
            return MaximalIndependentSetResult(
                decision="NOT_INDEPENDENT",
                blocking_edge=edge,
            )

    adjacency: list[set[int]] = [set() for _ in range(graph.vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    for vertex in range(graph.vertex_count):
        if vertex not in candidate and adjacency[vertex].isdisjoint(candidate):
            return MaximalIndependentSetResult(
                decision="INDEPENDENT_NOT_MAXIMAL",
                addable_vertex=vertex,
            )
    return MaximalIndependentSetResult(decision="MAXIMAL")


def edge_k_colorability(
    graph: SimpleUndirectedGraph, colors: int, solver_conflicts: int
) -> EdgeKColorabilityResult:
    """Decide whether a simple graph admits a proper ``k``-edge-coloring.

    A proper edge coloring assigns each edge a color in ``0..k-1`` such that
    incident edges receive distinct colors.  Uses a Z3 SAT search bounded by
    the caller-visible ``solver_conflicts`` budget and returns one proper
    coloring as a canonical source-bound value accepted directly by
    ``graph.edge_coloring.check``.  Non-colorability is claimed only on an
    explicit unsatisfiable outcome; an exhausted budget yields the typed
    ``SOLVER_BUDGET_EXCEEDED`` outcome instead of an unbounded wait.  The
    conflict budget bounds the SAT search after owner-local formula admission.
    """
    _admit_edge_coloring_graph(graph)
    _admit_solver_parameters(colors, solver_conflicts)
    edges = graph.edges

    def _colorable_result(witness: tuple[int, ...]) -> EdgeKColorabilityResult:
        return EdgeKColorabilityResult._from_kernel(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=EdgeColoringAssignment(
                graph=graph,
                colors=colors,
                coloring=witness,
            ),
        )

    if not edges:
        return _colorable_result(())
    outcome, coloring = run_coloring_worker("edge", graph, colors, solver_conflicts)
    if outcome == "sat":
        if coloring is None:
            raise AssertionError(
                "the bounded solver returned a satisfying outcome without a witness"
            )
        return _colorable_result(coloring)
    if outcome == "unsat":
        return EdgeKColorabilityResult._from_kernel(
            graph=graph,
            colors=colors,
            solver_conflicts=solver_conflicts,
            status="DECIDED",
            colorable=False,
            coloring=None,
        )
    return EdgeKColorabilityResult._from_kernel(
        graph=graph,
        colors=colors,
        solver_conflicts=solver_conflicts,
        status=(
            "SOLVER_BUDGET_EXCEEDED"
            if outcome == "budget_exceeded"
            else "EXECUTION_FAILED"
        ),
        colorable=None,
        coloring=None,
    )


def edge_coloring_check(
    assignment: EdgeColoringAssignment,
) -> EdgeColoringCheckResult:
    """Validate one source-bound edge-to-color assignment as a proper coloring."""
    _admit_edge_coloring_graph(assignment.graph)
    edges = assignment.graph.edges
    coloring = assignment.coloring
    for a, b in _incident_edge_index_pairs_for_canonical_graph(assignment.graph):
        if coloring[a] == coloring[b]:
            return EdgeColoringCheckResult._from_kernel(
                assignment=assignment,
                proper=False,
                blocking_edge=edges[a],
                conflicting_edge=edges[b],
            )
    return EdgeColoringCheckResult._from_kernel(
        assignment=assignment,
        proper=True,
        blocking_edge=None,
        conflicting_edge=None,
    )


__all__ = [
    "chromatic_number_certificate",
    "edge_coloring_check",
    "edge_k_colorability",
    "k_colorability",
    "maximal_independent_set",
]
