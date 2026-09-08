"""Killable owner for the chromatic-bipartition complete-search worker."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    ChromaticBipartitionResult,
    _unknown_result,
    _unordered_partition_count,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_BIPARTITION_WORKER = Path(__file__).with_name("_chromatic_bipartition_worker.py")
_WORKER_ERROR_BYTES = 16_384
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _chromatic_bipartition_worker_stdout_limit(
    request: ChromaticBipartitionRequest,
) -> int:
    """Measure the largest admitted canonical worker result for this request."""

    vertices = request.graph.vertices
    if len(vertices) < 2:
        result = ChromaticBipartitionResult(
            graph=request.graph,
            s=request.s,
            t=request.t,
            status="NO_SPLIT",
            checked_partitions=0,
        )
    else:
        result = ChromaticBipartitionResult(
            graph=request.graph,
            s=request.s,
            t=request.t,
            status="SPLIT",
            side_a=(vertices[0],),
            side_b=vertices[1:],
            chromatic_a=request.s,
            chromatic_b=request.t,
            checked_partitions=_unordered_partition_count(len(vertices)),
        )
    return len(
        json.dumps(
            result.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def find_chromatic_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Run the aggregate search in a killable worker with one request deadline."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return find_chromatic_bipartition(request)
    deadline = execution.started_at + request.resource_budget.wall_seconds
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    try:
        with TemporaryDirectory(prefix="jacobian-graph-bipartition-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _unknown_result(request)
            stdout_limit = _chromatic_bipartition_worker_stdout_limit(request)
            completed = run_bounded_process(
                [sys.executable, str(_BIPARTITION_WORKER)],
                input_bytes=json.dumps(
                    request.model_dump(mode="json"),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.resource_budget.wall_seconds)),
                    address_space_bytes=1_536 * 1024 * 1024,
                    file_size_bytes=max(_WORKER_FILE_SIZE_BYTES, stdout_limit),
                ),
                cwd=directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded chromatic bipartition worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError("chromatic bipartition worker cancelled")
    if completed.timed_out:
        return _unknown_result(request)
    request_checkpoint("after chromatic bipartition worker")
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise RuntimeError(
            "bounded chromatic bipartition worker exceeded an output cap"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "bounded chromatic bipartition worker did not establish an outcome"
        )
    if time.monotonic() >= deadline:
        return _unknown_result(request)
    try:
        result = ChromaticBipartitionResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded chromatic bipartition worker returned malformed output"
        ) from exc
    if result.graph != request.graph or result.s != request.s or result.t != request.t:
        raise RuntimeError(
            "chromatic bipartition worker result is not bound to the submitted request"
        )
    request_checkpoint("after chromatic bipartition response validation")
    return result
