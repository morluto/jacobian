"""Killable envelope for exact binomial and permutation construction."""

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
    request_checkpoint,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_COUNTING_WORKER = Path(__file__).resolve().with_name("_counting_worker.py")
_COUNTING_WALL_SECONDS = 120.0
_COUNTING_STDERR_LIMIT = 16_384
_COUNTING_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_COUNTING_FILE_SIZE_BYTES = 1024 * 1024


def _decode_count_result(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not value.isdigit()
    ):
        raise ValueError("counting worker returned a malformed exact count")
    return value


def _counting_stdout_limit(n: int, k: int) -> int:
    """Bound one decimal count from its admitted operands."""

    # Both nCk and nPk are at most n**k. The extra byte covers the inclusive
    # power-of-ten boundary; zero and empty products still need one digit.
    # Checked worker framing adds a fixed JSON envelope and trailing newline.
    return max(1, k * len(str(n)) + 1) + 64


def evaluate_count(operation: str, n: int, k: int) -> str:
    """Return the canonical decimal of one admitted comb or perm value."""

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = (
        execution.deadline
        if execution is not None and execution.deadline is not None
        else started + _COUNTING_WALL_SECONDS
    )
    bind_request_deadline(deadline)
    request_checkpoint("before exact counting")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before exact counting"
        )

    try:
        with TemporaryDirectory(prefix="jacobian-counting-") as worker_directory:
            result = run_checked_worker_process(
                [sys.executable, str(_COUNTING_WORKER)],
                input_bytes=json.dumps(
                    {"op": operation, "n": n, "k": k},
                    separators=(",", ":"),
                ).encode("utf-8"),
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_counting_stdout_limit(n, k),
                stderr_limit=_COUNTING_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(_COUNTING_WALL_SECONDS)),
                    address_space_bytes=_COUNTING_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_COUNTING_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
                decode_result=_decode_count_result,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("bounded counting worker could not be started") from exc

    request_checkpoint("after exact counting worker")
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired during exact counting"
        )
    return result


__all__ = ["evaluate_count"]
