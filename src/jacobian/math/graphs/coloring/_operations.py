"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.coloring._chromatic_number_models import (
    ChromaticNumberCertificateCheckRequest,
    ChromaticNumberCertificateCheckResult,
    _evaluate_chromatic_number_certificate,
)
from jacobian.math.graphs.coloring._models import (
    EdgeColoringAssignment,
    EdgeColoringCheckRequest,
    EdgeColoringCheckResult,
    EdgeKColorabilityRequest,
    EdgeKColorabilityResult,
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
    _incident_edge_index_pairs_for_canonical_graph,
)


def compute_chromatic_number_certificate_check(
    request: ChromaticNumberCertificateCheckRequest,
) -> ChromaticNumberCertificateCheckResult:
    """Check a proper coloring and exact fractional-clique lower certificate."""
    evaluation = _evaluate_chromatic_number_certificate(
        request.graph,
        request.claimed_chromatic_number,
        request.coloring,
        request.weights,
    )
    return ChromaticNumberCertificateCheckResult(
        graph=request.graph,
        claimed_chromatic_number=request.claimed_chromatic_number,
        coloring=request.coloring,
        weights=request.weights,
        verdict=evaluation.verdict,
        reason=evaluation.reason,
        weight_sum=CanonicalRational.from_fraction(evaluation.weight_sum),
        certified_lower_bound=evaluation.certified_lower_bound,
        blocking_vertex=evaluation.blocking_vertex,
        blocking_edge=evaluation.blocking_edge,
        blocking_independent_set=evaluation.blocking_independent_set,
        blocking_independent_set_weight=(
            None
            if evaluation.blocking_independent_set_weight is None
            else CanonicalRational.from_fraction(
                evaluation.blocking_independent_set_weight
            )
        ),
    )


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    colors = [z3.Int(f"color_{vertex}") for vertex in range(request.graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < request.colors) for color in colors))
    solver.add(*(colors[u] != colors[v] for u, v in request.graph.edges))
    if solver.check() == z3.sat:
        model = solver.model()
        coloring = tuple(model.eval(color).as_long() for color in colors)
        return KColorabilityResult(
            colorable=True,
            coloring=coloring,
            vertex_count=request.graph.vertex_count,
            colors=request.colors,
        )
    return KColorabilityResult(
        colorable=False,
        vertex_count=request.graph.vertex_count,
        colors=request.colors,
    )


def compute_maximal_independent_set_decision(
    request: MaximalIndependentSetRequest,
) -> MaximalIndependentSetResult:
    """Decide maximal independence and return the first canonical obstruction."""
    candidate = frozenset(request.candidate_set)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in request.graph.edges))
    for edge in edges:
        if edge[0] in candidate and edge[1] in candidate:
            return MaximalIndependentSetResult(
                decision="NOT_INDEPENDENT",
                blocking_edge=edge,
            )

    adjacency: list[set[int]] = [set() for _ in range(request.graph.vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    for vertex in range(request.graph.vertex_count):
        if vertex not in candidate and adjacency[vertex].isdisjoint(candidate):
            return MaximalIndependentSetResult(
                decision="INDEPENDENT_NOT_MAXIMAL",
                addable_vertex=vertex,
            )
    return MaximalIndependentSetResult(decision="MAXIMAL")


def compute_edge_k_colorability(
    request: EdgeKColorabilityRequest,
) -> EdgeKColorabilityResult:
    """Decide whether a simple graph admits a proper ``k``-edge-coloring.

    A proper edge coloring assigns each edge a color in ``0..k-1`` such that
    incident edges receive distinct colors.  Uses a Z3 SAT search bounded by
    the request-visible ``solver_conflicts`` budget and returns one proper
    coloring as a canonical source-bound value accepted directly by
    ``graph.edge_coloring.check``.  Non-colorability is claimed only on an
    explicit unsatisfiable outcome; an exhausted budget yields the typed
    ``SOLVER_BUDGET_EXCEEDED`` outcome instead of an unbounded wait.  The
    declared budget covers the whole request: decided-negative and
    budget-exceeded outcomes reuse the producing solve directly instead of
    paying a second replay, while independently supplied results still
    validate through full replay.
    """
    from jacobian.math.graphs.coloring._models import (
        _budget_exceeded_result,
        _decided_unsat_result,
        _run_edge_coloring_solver,
    )

    edges = request.graph.edges

    def _colorable_result(witness: tuple[int, ...]) -> EdgeKColorabilityResult:
        return EdgeKColorabilityResult(
            graph=request.graph,
            colors=request.colors,
            solver_conflicts=request.solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=EdgeColoringAssignment(
                graph=request.graph,
                colors=request.colors,
                coloring=witness,
            ),
            edge_count=len(edges),
        )

    if not edges:
        return _colorable_result(())
    outcome, coloring = _run_edge_coloring_solver(
        request.graph, request.colors, request.solver_conflicts
    )
    if outcome == "sat":
        if coloring is None:
            raise AssertionError(
                "the bounded solver returned a satisfying outcome without a witness"
            )
        return _colorable_result(coloring)
    if outcome == "unsat":
        return _decided_unsat_result(
            request.graph, request.colors, request.solver_conflicts
        )
    return _budget_exceeded_result(
        request.graph, request.colors, request.solver_conflicts
    )


def compute_edge_coloring_check(
    request: EdgeColoringCheckRequest,
) -> EdgeColoringCheckResult:
    """Validate one source-bound edge-to-color assignment as a proper coloring."""
    edges = request.assignment.graph.edges
    coloring = request.assignment.coloring
    for a, b in _incident_edge_index_pairs_for_canonical_graph(
        request.assignment.graph
    ):
        if coloring[a] == coloring[b]:
            return EdgeColoringCheckResult(
                assignment=request.assignment,
                proper=False,
                blocking_edge=edges[a],
                conflicting_edge=edges[b],
            )
    return EdgeColoringCheckResult(
        assignment=request.assignment,
        proper=True,
        blocking_edge=None,
        conflicting_edge=None,
    )
