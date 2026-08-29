"""Isolated Z3 adapter for one bounded graph-coloring decision."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs.coloring._coloring_process import (
    run_edge_coloring_solver_kernel,
    run_k_colorability_solver_kernel,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        kind = payload["kind"]
        colors = payload["colors"]
        solver_conflicts = payload["solver_conflicts"]
        if (
            kind not in {"vertex", "edge"}
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
        else:
            edge_graph = SimpleUndirectedGraph.model_validate(payload["graph"])
            outcome, coloring = run_edge_coloring_solver_kernel(
                edge_graph, colors, solver_conflicts
            )
        sys.stdout.write(
            json.dumps(
                {"outcome": outcome, "coloring": coloring},
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
