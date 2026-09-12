"""Deadline-bounded Gaussian Laurent GCD cancellation."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian import process
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import CanonicalLimits, encode_strict_json, loads_strict_json

_WORKER_PATH = Path(__file__).resolve().with_name("_laurent_gcd_worker.py")
_STDOUT_BYTES = 256 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024


def cancel_common_factor(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one Laurent GCD in a killable worker sharing the request deadline."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(monotonic()):
            return cancel_common_factor(payload)
    deadline = execution.deadline
    if deadline is None:
        deadline = execution.started_at + 120.0
    request_checkpoint("before trigonometric Laurent GCD encoding")
    encoded = encode_strict_json(payload)
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "trigonometric Laurent GCD deadline expired before the worker started"
        )
    try:
        with TemporaryDirectory(prefix="jacobian-trig-laurent-gcd-") as worker_dir:
            completed = process.run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=encoded,
                timeout_seconds=remaining,
                environment=process.worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_BYTES,
                stderr_limit=_STDERR_BYTES,
                resource_limits=process.ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_STDOUT_BYTES,
                ),
                cwd=worker_dir,
            )
    except OSError as exc:
        raise RuntimeError("bounded trigonometric Laurent GCD worker could not start") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "trigonometric Laurent GCD cancelled during the worker"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "trigonometric Laurent GCD deadline expired during the worker"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded trigonometric Laurent GCD worker did not establish a result")
    request_checkpoint("after trigonometric Laurent GCD")
    response = loads_strict_json(
        completed.stdout,
        limits=CanonicalLimits(
            max_input_bytes=_STDOUT_BYTES,
            max_output_bytes=_STDOUT_BYTES,
        ),
    )
    if not isinstance(response, dict):
        raise RuntimeError("bounded trigonometric Laurent GCD worker returned malformed output")
    return response
