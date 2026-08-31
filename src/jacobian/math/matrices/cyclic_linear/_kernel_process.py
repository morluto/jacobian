"""Bounded process owner for one rational cyclotomic kernel call."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.process import run_bounded_process, worker_environment

_CYCLIC_PROFILE_WALL_SECONDS = 3_600.0
_CYCLIC_KERNEL_WORKER = Path(__file__).resolve().with_name("_kernel_worker.py")
_CYCLIC_KERNEL_WORKER_STDOUT_BYTES = 64 * 1024 * 1024
_CYCLIC_KERNEL_WORKER_STDERR_BYTES = 64 * 1024


def run_cyclotomic_kernel(
    order: int,
    degree: int,
    matrix_coordinates: Any,
    common_denominator: int,
    deadline: float | None,
) -> tuple[Any, ...]:
    """Run one exact kernel in a bounded child process and decode its result."""

    input_data = pickle.dumps((order, degree, matrix_coordinates, common_denominator))
    remaining = (
        deadline - time.monotonic()
        if deadline is not None
        else _CYCLIC_PROFILE_WALL_SECONDS
    )
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "cyclotomic kernel subprocess exceeded the wall-time limit"
        )

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
        return pickle.loads(completed.stdout)  # type: ignore[no-any-return]
    except (EOFError, pickle.UnpicklingError, TypeError) as exc:
        raise RuntimeError("bounded cyclotomic kernel returned malformed output") from exc


__all__ = ["run_cyclotomic_kernel"]
