"""Bounded exact chromatic-number operation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    execution_deadline,
    request_checkpoint,
    require_execution_deadline,
)
from jacobian._worker_errors import decode_worker_execution_error
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.optimization._chromatic_kernel import (
    build_simple_graph,
    solve_chromatic_number,
)
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberOutput,
    GraphChromaticNumberRequest,
)
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
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


def _search_chromatic_number(
    request: GraphChromaticNumberRequest,
) -> GraphChromaticNumberOutput:
    """Run the complete Z3 chromatic search in a bounded owner worker."""

    deadline = execution_deadline(request.resource_budget.wall_seconds)
    try:
        with TemporaryDirectory(prefix="jacobian-graph-chromatic-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError("operation deadline expired")
            completed = run_bounded_process(
                [sys.executable, str(_CHROMATIC_NUMBER_WORKER)],
                input_bytes=json.dumps(
                    {"_deadline": deadline, **request.model_dump(mode="json")},
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
    except OSError as exc:
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    check_bounded_process_result(completed)
    require_execution_deadline(deadline)
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        require_execution_deadline(deadline)
        decode_worker_execution_error(response)
        result = GraphChromaticNumberOutput.model_validate(
            {
                **response,
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
            require_execution_deadline(deadline)
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        request_checkpoint("during chromatic-number response validation")
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc
    request_checkpoint("after chromatic-number response validation")
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError("operation deadline expired")
    return result


CHROMATIC_NUMBER_OPERATION = MathTool(
    operation_id="graph.invariant.chromatic_number.compute",
    title="Exact chromatic number",
    description=(
        "Compute the exact chromatic number of a bounded simple undirected "
        "graph by bounded Z3 k-colorability decisions. An incomplete search can return "
        "an UNKNOWN result with established bounds and the search trace."
        " Worker failures and parent deadline expiry raise execution errors; a valid partial result must arrive before that deadline."
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
        OperationExample(
            name="triangle_chromatic_number",
            description="Compute a triangle's chromatic number (3); vertices must be unique and edges must not self-loop.",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                }
            },
        ),
    ),
)
