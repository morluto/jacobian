"""Bounded process owner for the maximum-cut Z3 acceleration."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    lease_operation_phases,
    request_checkpoint,
    request_execution,
)
from jacobian.math.graphs.optimization._maximum_cut import (
    GraphMaximumCutRequest,
    GraphMaximumCutResult,
    _compute_maximum_cut_without_z3,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_MAXIMUM_CUT_WORKER = Path(__file__).with_name("_maximum_cut_worker.py")
_MAXIMUM_CUT_WORKER_WALL_SECONDS = 120
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _maximum_cut_worker_stdout_limit(request: GraphMaximumCutRequest) -> int:
    """Measure the largest source-bound projection for this admitted graph."""

    graph = request.graph
    projection = {
        "graph": graph.model_dump(mode="json"),
        "left_vertices": list(graph.vertices),
        "right_vertices": [],
        "crossing_edges": [list(edge) for edge in graph.edges],
        "cut_value": 32_640,
        "lower_bound": 32_640,
        "upper_bound": 32_640,
    }
    return len(
        json.dumps(projection, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )


def compute_maximum_cut_isolated(
    request: GraphMaximumCutRequest,
) -> GraphMaximumCutResult:
    """Run Z3 and the exact fallback under one request-owned deadline."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return compute_maximum_cut_isolated(request)
    stdout_limit = _maximum_cut_worker_stdout_limit(request)
    lease = lease_operation_phases(
        _MAXIMUM_CUT_WORKER_WALL_SECONDS,
        admitted_response_bytes=stdout_limit,
        validation_work=len(request.graph.vertices) + len(request.graph.edges),
    )
    request_checkpoint("before maximum-cut acceleration")
    try:
        with TemporaryDirectory(prefix="jacobian-maximum-cut-") as directory:
            payload = json.dumps(
                request.model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            remaining_seconds = lease.backend_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError(
                    "maximum-cut deadline expired before worker startup"
                )
            completed = run_bounded_process(
                [sys.executable, str(_MAXIMUM_CUT_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_MAXIMUM_CUT_WORKER_WALL_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        request_checkpoint("after maximum-cut worker startup failure")
        result = _compute_maximum_cut_without_z3(request)
        request_checkpoint("after maximum-cut exhaustive fallback")
        return result
    request_checkpoint("after maximum-cut acceleration")
    if completed.cancelled:
        raise OperationExecutionCancelledError("maximum-cut worker cancelled")
    if completed.timed_out:
        request_checkpoint("after maximum-cut worker timeout")
        result = _compute_maximum_cut_without_z3(request)
        request_checkpoint("after maximum-cut exhaustive fallback")
        return result
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        request_checkpoint("before maximum-cut exhaustive fallback")
        result = _compute_maximum_cut_without_z3(request)
        request_checkpoint("after maximum-cut exhaustive fallback")
        return result
    request_checkpoint("before maximum-cut response decoding")
    try:
        result = GraphMaximumCutResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
        if result.graph != request.graph:
            raise ValueError("worker result is not bound to the submitted graph")
        request_checkpoint("after maximum-cut response validation")
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        request_checkpoint("during maximum-cut response validation")
        result = _compute_maximum_cut_without_z3(request)
        request_checkpoint("after maximum-cut exhaustive fallback")
        return result
