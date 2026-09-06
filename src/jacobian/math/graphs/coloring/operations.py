"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError
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
    ColorCapacity,
    EdgeColoringAssignment,
    EdgeColoringCheckResult,
    EdgeColorList,
    EdgeKColorabilityResult,
    KColorabilityResult,
    ListCapacityEdgeColoringResult,
    ListEdgeColoringStatus,
    MaximalIndependentSetResult,
    VertexColoringAssignment,
    _incident_edge_index_pairs_for_canonical_graph,
    _require_edge_coloring_graph_bound,
    _require_indexed_coloring_graph,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)

if TYPE_CHECKING:
    from jacobian.math.combinatorics.exact_cover import GeneralizedExactCoverInstance


def _admit_chromatic_number_certificate(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> None:
    """Admit exact certificate work and retained-result bounds."""
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
    unsatisfiable outcome; an exhausted budget raises an operational timeout. The
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
    if outcome == "budget_exceeded":
        raise OperationExecutionTimeoutError(
            "vertex-coloring solver exhausted its conflict budget"
        )
    raise RuntimeError("vertex-coloring solver failed")


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
                graph=graph,
                candidate_set=candidate_set,
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
                graph=graph,
                candidate_set=candidate_set,
                decision="INDEPENDENT_NOT_MAXIMAL",
                addable_vertex=vertex,
            )
    return MaximalIndependentSetResult(
        graph=graph, candidate_set=candidate_set, decision="MAXIMAL"
    )


def verify_vertex_coloring(assignment: VertexColoringAssignment) -> bool:
    """Return whether a source-bound vertex coloring is proper."""
    return all(
        assignment.coloring[left] != assignment.coloring[right]
        for left, right in assignment.graph.edges
    )


vertex_coloring_check = verify_vertex_coloring


def verify_k_colorability(claim: KColorabilityResult) -> bool:
    """Verify the checkable positive part of a serialized colorability claim."""
    if not claim.colorable or claim.coloring is None:
        return False
    return (
        claim.coloring.graph == claim.graph
        and claim.coloring.colors == claim.colors
        and verify_vertex_coloring(claim.coloring)
    )


def verify_maximal_independent_set(claim: MaximalIndependentSetResult) -> bool:
    """Verify a serialized independence decision and its optional obstruction."""
    candidate = set(claim.candidate_set)
    edges = {tuple(sorted(edge)) for edge in claim.graph.edges}
    independent = not any(left in candidate and right in candidate for left, right in edges)
    if claim.decision == "NOT_INDEPENDENT":
        return (
            not independent
            and claim.blocking_edge is not None
            and tuple(claim.blocking_edge) in edges
            and set(claim.blocking_edge).issubset(candidate)
            and claim.addable_vertex is None
        )
    if claim.decision == "INDEPENDENT_NOT_MAXIMAL":
        if not independent or claim.addable_vertex is None:
            return False
        vertex = claim.addable_vertex
        return vertex not in candidate and all(
            vertex not in edge for edge in edges if edge[0] in candidate or edge[1] in candidate
        )
    if not independent:
        return False
    return all(
        vertex in candidate
        or any(vertex in edge and (edge[0] in candidate or edge[1] in candidate) for edge in edges)
        for vertex in range(claim.graph.vertex_count)
    )


maximal_independent_set_check = verify_maximal_independent_set


def edge_k_colorability(
    graph: SimpleUndirectedGraph, colors: int, solver_conflicts: int
) -> EdgeKColorabilityResult:
    """Decide whether a simple graph admits a proper ``k``-edge-coloring.

    A proper edge coloring assigns each edge a color in ``0..k-1`` such that
    incident edges receive distinct colors.  Uses a Z3 SAT search bounded by
    the caller-visible ``solver_conflicts`` budget and returns one proper
    coloring as a canonical source-bound value accepted directly by
    ``graph.edge_coloring.check``.  Non-colorability is claimed only on an
    explicit unsatisfiable outcome; an exhausted budget raises an operational
    timeout. The
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
    if outcome == "budget_exceeded":
        raise OperationExecutionTimeoutError(
            "edge-coloring solver exhausted its conflict budget"
        )
    raise RuntimeError("edge-coloring solver failed")


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
    "list_capacity_edge_coloring",
    "maximal_independent_set",
    "maximal_independent_set_check",
    "verify_k_colorability",
    "verify_maximal_independent_set",
    "verify_vertex_coloring",
    "vertex_coloring_check",
]


def _list_capacity_instance(
    graph: SimpleUndirectedGraph,
    palette: tuple[str, ...],
    lists: tuple[EdgeColorList, ...],
    capacities: tuple[ColorCapacity, ...],
) -> GeneralizedExactCoverInstance:
    """Build the slot-expanded exact-cover instance for one coloring request.

    One primary item per graph edge; secondary items per vertex/color pair
    enforce properness, and one secondary slot per color copy enforces
    capacities (clamped to the edge count). Every row carries a namespaced
    item ID so the primary, vertex/color, and slot namespaces stay disjoint
    for arbitrary labels. Returns the instance or raises a domain error when
    the expansion exceeds the exact-cover representation envelope.
    """

    from jacobian.math.combinatorics.exact_cover import (
        MAX_EXACT_COVER_INCIDENCES,
        MAX_EXACT_COVER_ITEMS,
        MAX_EXACT_COVER_ROWS,
        ExactCoverRow,
        GeneralizedExactCoverInstance,
    )

    edge_count = len(graph.edges)
    capacity_of = {entry.color: min(entry.capacity, edge_count) for entry in capacities}
    list_of = {tuple(entry.edge): tuple(entry.colors) for entry in lists}
    color_index = {color: position for position, color in enumerate(palette)}

    primary = tuple(f"edge:{index}" for index in range(edge_count))
    rows: list[ExactCoverRow] = []
    for edge_index, edge in enumerate(graph.edges):
        allowed = list_of[tuple(edge)]
        left, right = edge
        for color in allowed:
            slots = capacity_of[color]
            for slot in range(slots):
                items = tuple(
                    sorted(
                        (
                            f"edge:{edge_index}",
                            f"vc:{len(left)}:{left}:{color}",
                            f"vc:{len(right)}:{right}:{color}",
                            f"slot:{color}:{slot}",
                        )
                    )
                )
                rows.append(
                    ExactCoverRow(
                        row_id=f"row:{edge_index}:{color_index[color]}:{slot}",
                        items=items,
                    )
                )
    rows.sort(key=lambda row: row.row_id)
    primary_set = set(primary)
    secondary: set[str] = set()
    for row in rows:
        secondary.update(item for item in row.items if item not in primary_set)
    instance = GeneralizedExactCoverInstance(
        primary_items=primary,
        secondary_items=tuple(sorted(secondary)),
        rows=tuple(rows),
    )
    if (
        len(instance.primary_items) + len(instance.secondary_items)
        > MAX_EXACT_COVER_ITEMS
        or len(instance.rows) > MAX_EXACT_COVER_ROWS
        or sum(len(row.items) for row in instance.rows) > MAX_EXACT_COVER_INCIDENCES
    ):
        raise OperationDomainValidationError(
            location=("graph", "palette"),
            code="graph.list_edge_coloring.encoding_bound",
            message=(
                "the slot-expanded list/capacity encoding exceeds the "
                "exact-cover representation envelope"
            ),
        )
    return instance


def _decode_list_capacity_rows(
    graph: SimpleUndirectedGraph,
    palette: tuple[str, ...],
    lists: tuple[EdgeColorList, ...],
    capacities: tuple[ColorCapacity, ...],
    selected_row_ids: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Decode selected rows to a per-edge assignment, checking every relation."""

    list_of = {tuple(entry.edge): set(entry.colors) for entry in lists}
    capacity_of = {entry.color: entry.capacity for entry in capacities}
    assignment: dict[int, str] = {}
    counts: dict[str, int] = dict.fromkeys(palette, 0)
    for row_id in selected_row_ids:
        try:
            _, edge_index, color_index, _ = row_id.split(":", 3)
            edge_position = int(edge_index)
            color = palette[int(color_index)]
        except (ValueError, IndexError):
            return None
        if edge_position in assignment:
            return None
        edge = graph.edges[edge_position]
        if color not in list_of[tuple(edge)]:
            return None
        assignment[edge_position] = color
        counts[color] += 1
    if set(assignment) != set(range(len(graph.edges))):
        return None
    incident: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for position, color in assignment.items():
        left, right = graph.edges[position]
        if color in incident[left] or color in incident[right]:
            return None
        incident[left].add(color)
        incident[right].add(color)
    if any(counts[color] > capacity_of[color] for color in palette):
        return None
    return tuple(assignment[index] for index in range(len(graph.edges)))


def list_capacity_edge_coloring(
    graph: SimpleUndirectedGraph,
    palette: tuple[str, ...],
    lists: tuple[EdgeColorList, ...],
    capacities: tuple[ColorCapacity, ...],
) -> ListCapacityEdgeColoringResult:
    """Find a proper edge coloring within lists and capacities, or rule it out.

    The slot-expanded generalized exact-cover encoding is a private kernel
    choice: primary items force one row per edge, vertex/color secondaries
    force properness, and distinguishable color slots force capacities.
    FEASIBLE carries a checked assignment; INFEASIBLE follows exhaustive
    search; UNKNOWN records a node-limit stop without a conclusion.
    """

    from jacobian.math.combinatorics.exact_cover import (
        MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
        find_generalized_exact_cover,
    )

    def outcome(
        status: ListEdgeColoringStatus,
        assignment: tuple[str, ...] | None = None,
    ) -> ListCapacityEdgeColoringResult:
        return ListCapacityEdgeColoringResult._from_kernel(
            graph=graph,
            palette=palette,
            lists=lists,
            capacities=capacities,
            status=status,
            assignment=assignment,
        )

    instance = _list_capacity_instance(graph, palette, lists, capacities)
    try:
        result = find_generalized_exact_cover(
            instance,
            search_node_limit=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
        )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.list_edge_coloring.search_bound",
            message=str(error),
        ) from error
    if result.status == "FOUND":
        if result.selected_row_ids is None:
            return outcome("UNKNOWN")
        assignment = _decode_list_capacity_rows(
            graph, palette, lists, capacities, result.selected_row_ids
        )
        if assignment is None:
            return outcome("UNKNOWN")
        return outcome("FEASIBLE", assignment)
    if result.status == "NO_COVER":
        return outcome("INFEASIBLE")
    return outcome("UNKNOWN")
