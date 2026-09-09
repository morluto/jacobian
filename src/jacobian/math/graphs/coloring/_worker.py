"""Isolated Z3 adapter for one bounded graph-coloring decision."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationResourceExhaustedError,
    request_execution,
)
from jacobian._worker_errors import bind_worker_deadline, worker_execution_errors
from jacobian.math.graphs.coloring._coloring_process import (
    run_edge_coloring_solver_kernel,
    run_k_colorability_solver_kernel,
    run_precoloring_edge_repair_solver_kernel,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)
from jacobian.process import encode_worker_result_frame


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        kind = payload["kind"]
        colors = payload["colors"]
        solver_conflicts = payload["solver_conflicts"]
        if (
            kind not in {"vertex", "edge", "precoloring_edge_repair"}
            or not isinstance(colors, int)
            or isinstance(colors, bool)
            or not isinstance(solver_conflicts, int)
            or isinstance(solver_conflicts, bool)
        ):
            raise ValueError("worker payload has invalid coloring inputs")
        if kind == "vertex":
            indexed_graph = IndexedSimpleUndirectedGraph.model_validate(
                payload["graph"]
            )
            outcome, coloring = run_k_colorability_solver_kernel(
                indexed_graph, colors, solver_conflicts
            )
        elif kind == "edge":
            edge_graph = SimpleUndirectedGraph.model_validate(payload["graph"])
            outcome, coloring = run_edge_coloring_solver_kernel(
                edge_graph, colors, solver_conflicts
            )
        else:
            indexed_graph = IndexedSimpleUndirectedGraph.model_validate(
                payload["graph"]
            )
            fixed_colors = tuple(
                (int(vertex), int(color)) for vertex, color in payload["fixed_colors"]
            )
            outcome, coloring = run_precoloring_edge_repair_solver_kernel(
                indexed_graph, colors, fixed_colors, solver_conflicts
            )
        if outcome == "budget_exceeded":
            raise OperationResourceExhaustedError(ExecutionResource.WORK)
        if outcome == "execution_failed":
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        sys.stdout.buffer.write(
            encode_worker_result_frame({"outcome": outcome, "coloring": coloring})
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
