"""Bounded exact chromatic-number operation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberOutput,
    GraphChromaticNumberRequest,
)
from jacobian.math.graphs.optimization._operations import (
    build_simple_graph,
    solve_chromatic_number,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_CHROMATIC_NUMBER_WORKER = Path(__file__).with_name("_chromatic_number_worker.py")
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _search_chromatic_number_kernel(
    request: GraphChromaticNumberRequest,
) -> GraphChromaticNumberOutput:
    """Run bounded k-colorability decisions until exactness or timeout."""

    started = time.monotonic()
    networkx_graph = build_simple_graph(request.graph)
    output = solve_chromatic_number(
        networkx_graph,
        graph=request.graph,
        vertices=request.graph.vertices,
        wall_seconds=request.resource_budget.wall_seconds,
        started=started,
    )

    return output


def _chromatic_worker_failure(
    request: GraphChromaticNumberRequest, detail: str
) -> GraphChromaticNumberOutput:
    """Return a source-derived unknown when the bounded worker fails."""

    vertices = request.graph.vertices
    if not vertices:
        return GraphChromaticNumberOutput(
            status="EXACT",
            vertices=vertices,
            order=0,
            chromatic_number=0,
            lower_bound=0,
            upper_bound=0,
            coloring={},
            solver_status="SPECIAL_CASE",
            tested=(),
            detail="the empty graph requires zero colors",
        )
    return GraphChromaticNumberOutput(
        status="UNKNOWN",
        vertices=vertices,
        order=len(vertices),
        lower_bound=2 if request.graph.edges else 1,
        upper_bound=len(vertices),
        coloring={vertex: index for index, vertex in enumerate(vertices)},
        solver_status="UNKNOWN",
        tested=(),
        detail=detail,
    )


def _search_chromatic_number(
    request: GraphChromaticNumberRequest,
) -> GraphChromaticNumberOutput:
    """Run the complete Z3 chromatic search in a bounded owner worker."""

    deadline = time.monotonic() + request.resource_budget.wall_seconds
    try:
        with TemporaryDirectory(prefix="jacobian-graph-chromatic-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _chromatic_worker_failure(
                    request,
                    "the chromatic-number request expired before worker startup",
                )
            completed = run_bounded_process(
                [sys.executable, str(_CHROMATIC_NUMBER_WORKER)],
                input_bytes=json.dumps(
                    request.model_dump(mode="json"),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.resource_budget.wall_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return _chromatic_worker_failure(
            request, "the bounded chromatic-number worker could not be started"
        )
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _chromatic_worker_failure(
            request, "the bounded chromatic-number worker did not establish an outcome"
        )
    if time.monotonic() >= deadline:
        return _chromatic_worker_failure(
            request, "the chromatic-number request expired before response validation"
        )
    try:
        result = GraphChromaticNumberOutput.model_validate(
            {
                **json.loads(completed.stdout.decode("utf-8")),
                "vertices": list(request.graph.vertices),
            }
        )
        if result.order != len(request.graph.vertices) or (
            result.coloring is not None
            and any(
                result.coloring[left] == result.coloring[right]
                for left, right in request.graph.edges
            )
        ):
            raise ValueError("worker result is not bound to the submitted graph")
        if time.monotonic() < deadline:
            return result
        raise ValueError("request expired during response validation")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _chromatic_worker_failure(
            request, "the bounded chromatic-number worker returned malformed output"
        )


CHROMATIC_NUMBER_OPERATION = MathTool(
    operation_id="graph.invariant.chromatic_number.compute",
    title="Exact chromatic number",
    description=(
        "Compute the exact chromatic number of a bounded simple undirected "
        "graph by bounded Z3 k-colorability decisions. A timeout returns "
        "an UNKNOWN result with the tested bounds and search trace."
    ),
    request_type=GraphChromaticNumberRequest,
    result_type=GraphChromaticNumberOutput,
    run=_search_chromatic_number,
    tags=(
        "graph",
        "invariant",
        "chromatic_number",
        "exact",
        "bounded",
        "z3",
    ),
    examples=(
        example(
            "triangle_chromatic_number",
            "Compute a triangle's chromatic number (3); vertices must be unique and edges must not self-loop.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                }
            },
        ),
    ),
)
