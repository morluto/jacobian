"""Bounded process owner for one rational cyclotomic kernel call."""

from __future__ import annotations

import pickle
import sys
import time
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)

_CYCLIC_PROFILE_WALL_SECONDS = 3_600.0
_CYCLIC_KERNEL_WORKER = Path(__file__).resolve().with_name("_kernel_worker.py")
_CYCLIC_KERNEL_WORKER_STDOUT_BYTES = 64 * 1024 * 1024
_CYCLIC_KERNEL_WORKER_STDERR_BYTES = 64 * 1024


def run_cyclotomic_kernels(
    requests: tuple[tuple[int, int, Any, int], ...],
    deadline: float | None,
) -> tuple[tuple[Any, ...], ...]:
    """Run all exact component kernels in one bounded child process."""

    input_data = pickle.dumps(requests)
    request_digest = sha256(input_data).digest()
    remaining = (
        deadline - time.monotonic()
        if deadline is not None
        else _CYCLIC_PROFILE_WALL_SECONDS
    )
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "cyclotomic kernel subprocess exceeded the wall-time limit"
        )

    from jacobian.process import run_bounded_process, worker_environment

    try:
        with TemporaryDirectory(prefix="jacobian-cyclic-kernel-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_CYCLIC_KERNEL_WORKER)],
                input_bytes=input_data,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_CYCLIC_KERNEL_WORKER_STDOUT_BYTES,
                stderr_limit=_CYCLIC_KERNEL_WORKER_STDERR_BYTES,
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError("bounded cyclotomic kernel could not be started") from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "cyclotomic kernel subprocess was cancelled"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "cyclotomic kernel subprocess exceeded the wall-time limit"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded cyclotomic kernel did not produce an exact result")
    try:
        response_digest, results = pickle.loads(completed.stdout)
        if response_digest != request_digest or not isinstance(results, tuple):
            raise TypeError("unbound cyclotomic kernel response")
        if len(results) != len(requests):
            raise TypeError("wrong cyclotomic kernel response count")
        return results
    except (EOFError, pickle.UnpicklingError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded cyclotomic kernel returned malformed output"
        ) from exc


__all__ = ["run_cyclotomic_kernels"]
