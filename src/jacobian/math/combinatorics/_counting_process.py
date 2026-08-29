"""Killable envelope for exact binomial and permutation construction."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import bind_request_deadline, current_request_execution
from jacobian.canonical import CanonicalLimits
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_COUNTING_WORKER = Path(__file__).resolve().with_name("_counting_worker.py")
_COUNTING_WALL_SECONDS = 120.0
_COUNTING_STDERR_LIMIT = 16_384
_COUNTING_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_COUNTING_FILE_SIZE_BYTES = 1024 * 1024


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
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("request deadline expired before exact counting")

    try:
        with TemporaryDirectory(prefix="jacobian-counting-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_COUNTING_WORKER)],
                input_bytes=json.dumps(
                    {"op": operation, "n": n, "k": k},
                    separators=(",", ":"),
                ).encode("utf-8"),
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=CanonicalLimits().max_output_bytes,
                stderr_limit=_COUNTING_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(_COUNTING_WALL_SECONDS)),
                    address_space_bytes=_COUNTING_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_COUNTING_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError("bounded counting worker could not be started") from exc

    if completed.cancelled:
        raise InterruptedError("request cancelled during exact counting")
    if completed.timed_out:
        raise TimeoutError("request deadline expired during exact counting")
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded counting worker did not establish an exact count")

    try:
        text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("bounded counting worker returned malformed output") from exc
    if not text or not text.isdigit():
        raise RuntimeError("bounded counting worker returned malformed output")
    return text


__all__ = ["evaluate_count"]
