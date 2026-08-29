"""Killable process boundary for bounded graph-coloring solver calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from jacobian.math.graphs.coloring._models import (
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

_COLORING_WORKER = Path(__file__).with_name("_worker.py")
_COLORING_WORKER_WALL_SECONDS = 30
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024
ColoringWorkerOutcome = Literal["sat", "unsat", "budget_exceeded", "execution_failed"]


def run_k_colorability_solver_kernel(
    graph: IndexedSimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[ColoringWorkerOutcome, tuple[int, ...] | None]:
    """Run one bounded vertex-coloring Z3 transaction."""

    import z3

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


def run_edge_coloring_solver_kernel(
    graph: SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[ColoringWorkerOutcome, tuple[int, ...] | None]:
    """Run one bounded edge-coloring Z3 transaction."""

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


def run_coloring_worker(
    kind: Literal["vertex", "edge"],
    graph: IndexedSimpleUndirectedGraph | SimpleUndirectedGraph,
    colors: int,
    solver_conflicts: int,
) -> tuple[ColoringWorkerOutcome, tuple[int, ...] | None]:
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


__all__ = [
    "run_coloring_worker",
    "run_edge_coloring_solver_kernel",
    "run_k_colorability_solver_kernel",
]
