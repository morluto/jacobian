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
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_BIPARTITION_WORKER = Path(__file__).with_name("_chromatic_bipartition_worker.py")
_WORKER_OUTPUT_BYTES = 128 * 1024
_WORKER_ERROR_BYTES = 16_384


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
            completed = run_bounded_process(
                [sys.executable, str(_BIPARTITION_WORKER)],
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
                    address_space_bytes=1_536 * 1024 * 1024,
                    file_size_bytes=1_024 * 1_024,
                ),
                cwd=directory,
            )
    except OSError:
        return _unknown_result(request)
    request_checkpoint("after chromatic bipartition worker")
    if completed.cancelled:
        raise OperationExecutionCancelledError("chromatic bipartition worker cancelled")
    if (
        completed.timed_out
        or completed.returncode != 0
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return _unknown_result(request)
    if time.monotonic() >= deadline:
        return _unknown_result(request)
    try:
        result = ChromaticBipartitionResult.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _unknown_result(request)
    request_checkpoint("after chromatic bipartition response validation")
    return result.model_copy(
        update={"graph": request.graph, "s": request.s, "t": request.t}
    )
