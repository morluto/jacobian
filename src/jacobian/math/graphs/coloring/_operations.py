"""Domain-owned graph coloring and independent set operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

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
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

# ``max_conflicts`` bounds only Z3's search.  Formula construction and model
# extraction remain executable work, so the owner also places the complete
# transaction in a killable process with a deliberately generous fallback
# envelope.  Its expiry carries no mathematical conclusion.
_COLORING_WORKER = Path(__file__).with_name("_worker.py")
_COLORING_WORKER_WALL_SECONDS = 30
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024
_ColoringWorkerOutcome = Literal["sat", "unsat", "budget_exceeded", "execution_failed"]


def _run_k_colorability_solver_kernel(
    graph: IndexedSimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[_ColoringWorkerOutcome, tuple[int, ...] | None]:
    """Run the existing bounded vertex-coloring Z3 adapter once."""

    import z3  # type: ignore[import-untyped]

    try:
        solver = z3.Solver()
        solver.set("max_conflicts", solver_conflicts)
        vertex_colors = [
            z3.Int(f"color_{vertex}") for vertex in range(graph.vertex_count)
        ]
        solver.add(*(z3.And(color >= 0, color < colors) for color in vertex_colors))
        solver.add(*(vertex_colors[u] != vertex_colors[v] for u, v in graph.edges))
        outcome = solver.check()
        if outcome == z3.sat:
            model = solver.model()
            return "sat", tuple(model.eval(color).as_long() for color in vertex_colors)
        if outcome == z3.unsat:
            return "unsat", None
        return (
            "budget_exceeded"
            if "max-conflicts-reached" in solver.reason_unknown()
            else "execution_failed",
            None,
        )
    except z3.Z3Exception:
        return "execution_failed", None


def _run_edge_coloring_solver_kernel(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[_ColoringWorkerOutcome, tuple[int, ...] | None]:
    """Run the existing bounded edge-coloring Z3 adapter once."""

    import z3

    try:
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
        return (
            "budget_exceeded"
            if "max-conflicts-reached" in solver.reason_unknown()
            else "execution_failed",
            None,
        )
    except z3.Z3Exception:
        return "execution_failed", None


def _run_coloring_worker(
    kind: Literal["vertex", "edge"],
    graph: IndexedSimpleUndirectedGraph | SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[_ColoringWorkerOutcome, tuple[int, ...] | None]:
    """Run one complete coloring solver transaction in an isolated worker."""

    try:
        with TemporaryDirectory(prefix="jacobian-graph-coloring-") as directory:
            completed = run_bounded_process(
                [sys.executable, str(_COLORING_WORKER)],
                input_bytes=json.dumps(
                    {
                        "kind": kind,
                        "graph": graph.model_dump(mode="json"),
                        "colors": colors,
                        "solver_conflicts": solver_conflicts,
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=_COLORING_WORKER_WALL_SECONDS,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=_COLORING_WORKER_WALL_SECONDS,
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return "execution_failed", None
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return "execution_failed", None
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        outcome = payload["outcome"]
        coloring = payload["coloring"]
        if outcome not in {"sat", "unsat", "budget_exceeded", "execution_failed"}:
            raise ValueError("worker returned an invalid solver outcome")
        if coloring is None:
            return outcome, None
        if not isinstance(coloring, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in coloring
        ):
            raise ValueError("worker returned an invalid coloring")
        return outcome, tuple(coloring)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "execution_failed", None


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
    return ChromaticNumberCertificateCheckResult._from_kernel(
        graph=request.graph,
        claimed_chromatic_number=request.claimed_chromatic_number,
        coloring=request.coloring,
        weights=request.weights,
        evaluation=evaluation,
    )


def verify_chromatic_number_certificate_check_result(
    result: ChromaticNumberCertificateCheckResult,
) -> bool:
    """Replay one independently supplied bounded chromatic certificate claim."""

    try:
        request = ChromaticNumberCertificateCheckRequest(
            graph=result.graph,
            claimed_chromatic_number=result.claimed_chromatic_number,
            coloring=result.coloring,
            weights=result.weights,
        )
    except ValueError:
        return False
    expected = _evaluate_chromatic_number_certificate(
        request.graph,
        request.claimed_chromatic_number,
        request.coloring,
        request.weights,
    )
    return (
        result.verdict,
        result.reason,
        result.weight_sum.as_fraction(),
        result.certified_lower_bound,
        result.blocking_vertex,
        result.blocking_edge,
        result.blocking_independent_set,
        None
        if result.blocking_independent_set_weight is None
        else result.blocking_independent_set_weight.as_fraction(),
    ) == (
        expected.verdict,
        expected.reason,
        expected.weight_sum,
        expected.certified_lower_bound,
        expected.blocking_vertex,
        expected.blocking_edge,
        expected.blocking_independent_set,
        expected.blocking_independent_set_weight,
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
    if not request.graph.edges:
        return KColorabilityResult._from_kernel(
            graph=request.graph,
            colors=request.colors,
            solver_conflicts=request.solver_conflicts,
            status="DECIDED",
            colorable=True,
            coloring=(0,) * request.graph.vertex_count,
        )
    outcome, coloring = _run_coloring_worker(
        "vertex", request.graph, request.colors, request.solver_conflicts
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
        status=(
            "SOLVER_BUDGET_EXCEEDED"
            if outcome == "budget_exceeded"
            else "EXECUTION_FAILED"
        ),
        colorable=None,
        coloring=None,
    )


def verify_k_colorability_result(result: KColorabilityResult) -> bool:
    """Replay only a separately supplied negative or incomplete SAT claim."""

    if result.status == "DECIDED" and result.colorable is True:
        return True
    outcome, _coloring = _run_coloring_worker(
        "vertex", result.graph, result.colors, result.solver_conflicts
    )
    if result.status == "SOLVER_BUDGET_EXCEEDED":
        return outcome == "budget_exceeded"
    if result.status == "EXECUTION_FAILED":
        return False
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
    outcome, coloring = _run_coloring_worker(
        "edge", request.graph, request.colors, request.solver_conflicts
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
        status=(
            "SOLVER_BUDGET_EXCEEDED"
            if outcome == "budget_exceeded"
            else "EXECUTION_FAILED"
        ),
        colorable=None,
        coloring=None,
    )


def verify_edge_k_colorability_result(result: EdgeKColorabilityResult) -> bool:
    """Replay only a separately supplied negative or incomplete SAT claim."""

    if result.status == "DECIDED" and result.colorable is True:
        return True
    outcome, _coloring = _run_coloring_worker(
        "edge", result.graph, result.colors, result.solver_conflicts
    )
    if result.status == "SOLVER_BUDGET_EXCEEDED":
        return outcome == "budget_exceeded"
    if result.status == "EXECUTION_FAILED":
        return False
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
