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
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def _run_k_colorability_solver(
    graph: IndexedSimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[str, tuple[int, ...] | None]:
    """Run the existing bounded vertex-coloring Z3 adapter once."""

    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    vertex_colors = [z3.Int(f"color_{vertex}") for vertex in range(graph.vertex_count)]
    solver.add(*(z3.And(color >= 0, color < colors) for color in vertex_colors))
    solver.add(*(vertex_colors[u] != vertex_colors[v] for u, v in graph.edges))
    outcome = solver.check()
    if outcome == z3.sat:
        model = solver.model()
        return "sat", tuple(model.eval(color).as_long() for color in vertex_colors)
    if outcome == z3.unsat:
        return "unsat", None
    return "unknown", None


def _run_edge_coloring_solver(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[str, tuple[int, ...] | None]:
    """Run the existing bounded edge-coloring Z3 adapter once."""

    import z3

    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    edge_colors = [z3.Int(f"c_{index}") for index in range(len(graph.edges))]
    solver.add(*(z3.And(color >= 0, color < colors) for color in edge_colors))
    for first, second in _incident_edge_index_pairs_for_canonical_graph(graph):
        solver.add(edge_colors[first] != edge_colors[second])
    outcome = solver.check()
    if outcome == z3.sat:
        model = solver.model()
        return "sat", tuple(model.eval(color).as_long() for color in edge_colors)
    if outcome == z3.unsat:
        return "unsat", None
    return "unknown", None


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
    """Decide whether a simple graph admits a proper ``k``-coloring.

    Uses a Z3 SAT search bounded by the request-visible ``solver_conflicts``
    budget and returns one proper coloring as the witness of a colorable
    decision.  Non-colorability is claimed only on an explicit
    unsatisfiable outcome; an exhausted budget yields the typed
    ``SOLVER_BUDGET_EXCEEDED`` outcome instead of an unbounded wait.  The
    conflict budget bounds the SAT search after owner-local formula admission;
    separately supplied negative or incomplete claims may be replayed through
    the explicit verifier.
    """
    outcome, coloring = _run_k_colorability_solver(
        request.graph, request.colors, request.solver_conflicts
    )
    if outcome == "sat":
        if coloring is None:
            raise AssertionError(
                "the bounded solver returned a satisfying outcome without a witness"
            )
        return KColorabilityResult._from_kernel(
            graph=request.graph,
            colors=request.colors,
            solver_conflicts=request.solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=coloring,
        )
    if outcome == "unsat":
        return KColorabilityResult._from_kernel(
            graph=request.graph,
            colors=request.colors,
            solver_conflicts=request.solver_conflicts,
            status="DECIDED",
            colorable=False,
            coloring=None,
        )
    return KColorabilityResult._from_kernel(
        graph=request.graph,
        colors=request.colors,
        solver_conflicts=request.solver_conflicts,
        status="SOLVER_BUDGET_EXCEEDED",
        colorable=None,
        coloring=None,
    )


def verify_k_colorability_result(result: KColorabilityResult) -> bool:
    """Replay only a separately supplied negative or incomplete SAT claim."""

    if result.status == "DECIDED" and result.colorable is True:
        return True
    outcome, _coloring = _run_k_colorability_solver(
        result.graph, result.colors, result.solver_conflicts
    )
    if result.status == "SOLVER_BUDGET_EXCEEDED":
        return outcome == "unknown"
    return outcome == "unsat"


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
    conflict budget bounds the SAT search after owner-local formula admission;
    separately supplied negative or incomplete claims may be replayed through
    the explicit verifier.
    """
    edges = request.graph.edges

    def _colorable_result(witness: tuple[int, ...]) -> EdgeKColorabilityResult:
        return EdgeKColorabilityResult._from_kernel(
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
        return EdgeKColorabilityResult._from_kernel(
            graph=request.graph,
            colors=request.colors,
            solver_conflicts=request.solver_conflicts,
            status="DECIDED",
            colorable=False,
            coloring=None,
        )
    return EdgeKColorabilityResult._from_kernel(
        graph=request.graph,
        colors=request.colors,
        solver_conflicts=request.solver_conflicts,
        status="SOLVER_BUDGET_EXCEEDED",
        colorable=None,
        coloring=None,
    )


def verify_edge_k_colorability_result(result: EdgeKColorabilityResult) -> bool:
    """Replay only a separately supplied negative or incomplete SAT claim."""

    if result.status == "DECIDED" and result.colorable is True:
        return True
    outcome, _coloring = _run_edge_coloring_solver(
        result.graph, result.colors, result.solver_conflicts
    )
    if result.status == "SOLVER_BUDGET_EXCEEDED":
        return outcome == "unknown"
    return outcome == "unsat"


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
