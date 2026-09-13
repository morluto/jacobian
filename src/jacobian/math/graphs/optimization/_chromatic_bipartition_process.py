"""Killable owner for the chromatic-bipartition complete-search worker."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    current_request_execution,
    lease_operation_phases,
    report_request_progress,
    request_checkpoint,
    request_execution,
    require_execution_deadline,
)
from jacobian._worker_protocol import (
    encode_worker_result_frame,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    MAX_CHROMATIC_BIPARTITION_PARTITIONS,
    ChromaticBipartitionRequest,
    ChromaticBipartitionResult,
    _chromatic_bipartition_can_return_split,
    _unordered_partition_count,
)
from jacobian.process import (
    ProcessResourceLimits,
    decode_checked_worker_output,
    run_bounded_process,
    worker_environment,
)

_BIPARTITION_WORKER = Path(__file__).with_name("_chromatic_bipartition_worker.py")
_WORKER_ERROR_BYTES = 16_384
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _serialized_result_bytes(result: ChromaticBipartitionResult) -> int:
    return len(encode_worker_result_frame(result.model_dump(mode="json")))


def _chromatic_bipartition_worker_stdout_limit(
    request: ChromaticBipartitionRequest,
) -> int:
    """Measure the largest admitted canonical worker result for this request."""

    vertices = request.graph.vertices
    checked = min(
        _unordered_partition_count(len(vertices)),
        MAX_CHROMATIC_BIPARTITION_PARTITIONS,
    )
    envelopes = [
        ChromaticBipartitionResult(
            graph=request.graph,
            s=request.s,
            t=request.t,
            status="NO_SPLIT",
            checked_partitions=checked,
        )
    ]
    if _chromatic_bipartition_can_return_split(request):
        envelopes.append(
            ChromaticBipartitionResult(
                graph=request.graph,
                s=request.s,
                t=request.t,
                status="SPLIT",
                side_a=(vertices[0],),
                side_b=vertices[1:],
                chromatic_a=request.s,
                chromatic_b=request.t,
                checked_partitions=checked,
            )
        )
    return max(_serialized_result_bytes(result) for result in envelopes)


def _chromatic_bipartition_timeout(
    message: str, request: ChromaticBipartitionRequest
) -> None:
    raise OperationExecutionTimeoutError(
        message,
        configured_seconds=request.resource_budget.wall_seconds,
        adjustable_field_path=("resource_budget", "wall_seconds"),
    )


def find_chromatic_bipartition(
    request: ChromaticBipartitionRequest,
) -> ChromaticBipartitionResult:
    """Run the aggregate search in a killable worker with one request deadline."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return find_chromatic_bipartition(request)
    total_partitions = _unordered_partition_count(len(request.graph.vertices))
    report_request_progress(
        0,
        total=total_partitions,
        message="chromatic bipartitions checked",
    )
    stdout_limit = _chromatic_bipartition_worker_stdout_limit(request)
    lease = lease_operation_phases(
        request.resource_budget.wall_seconds,
        admitted_response_bytes=stdout_limit,
        validation_work=len(request.graph.vertices) + len(request.graph.edges),
    )
    deadline = lease.operation_deadline
    try:
        with TemporaryDirectory(prefix="jacobian-graph-bipartition-") as directory:
            remaining_seconds = lease.backend_deadline - time.monotonic()
            if remaining_seconds <= 0:
                _chromatic_bipartition_timeout(
                    "chromatic bipartition deadline expired before the worker started",
                    request,
                )
            completed = run_bounded_process(
                [sys.executable, str(_BIPARTITION_WORKER)],
                input_bytes=json.dumps(
                    {
                        "_deadline": lease.backend_deadline,
                        **request.model_dump(mode="json"),
                    },
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
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError("chromatic bipartition worker cancelled")
    if completed.timed_out:
        _chromatic_bipartition_timeout(
            "chromatic bipartition deadline expired during the worker",
            request,
        )
    request_checkpoint("after chromatic bipartition worker")
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise OperationResourceExhaustedError(ExecutionResource.OUTPUT)
    if completed.returncode != 0:
        raise OperationBackendError(BackendFailureReason.ABNORMAL_EXIT)
    if time.monotonic() >= deadline:
        _chromatic_bipartition_timeout(
            "chromatic bipartition deadline expired after the worker returned",
            request,
        )
    try:
        result = decode_checked_worker_output(
            completed.stdout,
            decode_result=ChromaticBipartitionResult.model_validate,
            checkpoint=lambda: require_execution_deadline(deadline),
        )
    except (TypeError, ValueError) as exc:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc
    if result.graph != request.graph or result.s != request.s or result.t != request.t:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    request_checkpoint("after chromatic bipartition response validation")
    report_request_progress(
        result.checked_partitions,
        total=total_partitions,
        message="chromatic bipartitions checked",
    )
    return result
