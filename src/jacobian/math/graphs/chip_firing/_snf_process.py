"""Killable FLINT SNF process boundary for chip-firing critical groups."""

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
    bind_request_deadline,
    current_request_execution,
)

_SNF_WORKER = Path(__file__).resolve().with_name("_snf_worker.py")
# Admitted reduced Laplacians satisfy dimension**3 <= 1_500_000. FLINT's
# exact integer SNF on that envelope finishes well under a minute; keep a
# generous killable ceiling so one stalled backend cannot outlive the request.
_SNF_WALL_SECONDS = 120.0
_SNF_STDOUT_LIMIT = 256 * 1024
_SNF_STDERR_LIMIT = 64 * 1024
_SNF_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_SNF_FILE_SIZE_BYTES = 1024 * 1024


def smith_normal_form_diagonal(matrix: list[list[int]]) -> tuple[int, ...]:
    """Return the SNF diagonal through a deadline-bound killable FLINT worker."""

    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    if rows == 0 or cols == 0:
        return ()

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + _SNF_WALL_SECONDS
    bind_request_deadline(deadline)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before reduced-Laplacian SNF"
        )

    try:
        with TemporaryDirectory(prefix="jacobian-chip-firing-snf-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_SNF_WORKER)],
                input_bytes=json.dumps(
                    {"matrix": matrix}, separators=(",", ":")
                ).encode("utf-8"),
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_SNF_STDOUT_LIMIT,
                stderr_limit=_SNF_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(_SNF_WALL_SECONDS)),
                    address_space_bytes=_SNF_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_SNF_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded chip-firing SNF worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during reduced-Laplacian SNF"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during reduced-Laplacian SNF"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded chip-firing SNF worker did not establish a diagonal"
        )

    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        if response.get("ok") is not True:
            raise ValueError("worker reported failure")
        diagonal = response["diagonal"]
        if not isinstance(diagonal, list) or any(
            not isinstance(value, int) for value in diagonal
        ):
            raise ValueError("worker diagonal is malformed")
        return tuple(int(value) for value in diagonal)
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "bounded chip-firing SNF worker returned malformed output"
        ) from exc


__all__ = ["smith_normal_form_diagonal"]
