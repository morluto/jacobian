"""Bounded child-process owner for the root-critical profile kernel.

The kernel runs in a killable child so a wedged algebraic-number computation
cannot hold the request.  Process ownership lives here rather than in the
public operations module: the architecture contract requires every
``run_bounded_process`` call to sit in a module whose name states that external
responsibility.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    RequestCancellationSignal,
)
from jacobian.canonical import encode_strict_json
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

PROFILE_WORKER_PATH = Path(__file__).resolve().with_name("_profile_worker.py")
PROFILE_STDOUT_BYTES = 64 * 1024 * 1024
PROFILE_STDERR_BYTES = 64 * 1024
PROFILE_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024

__all__ = ["PROFILE_STDOUT_BYTES", "run_profile_worker_process"]


def run_profile_worker_process(
    polynomial: RationalPolynomial,
    *,
    max_pair_rows: object,
    remaining_seconds: float,
    cancellation_signal: RequestCancellationSignal | None,
) -> bytes:
    """Run the kernel worker once and return its validated stdout bytes.

    The caller owns the deadline arithmetic; this owner refuses a
    non-positive allowance, confines the child's cpu, address space, and
    stream limits, and translates the supervisor's outcome into the request's
    typed cancellation or deadline error.
    """

    if remaining_seconds <= 0:
        raise OperationExecutionTimeoutError(
            "root-critical profile deadline expired before the kernel worker"
        )
    payload = encode_strict_json(
        {
            "polynomial": polynomial.model_dump_json(),
            "max_pair_rows": max_pair_rows,
        }
    )
    try:
        completed = run_bounded_process(
            [sys.executable, str(PROFILE_WORKER_PATH)],
            input_bytes=payload,
            timeout_seconds=remaining_seconds,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=PROFILE_STDOUT_BYTES,
            stderr_limit=PROFILE_STDERR_BYTES,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(remaining_seconds)),
                address_space_bytes=PROFILE_ADDRESS_SPACE_BYTES,
                file_size_bytes=PROFILE_STDOUT_BYTES,
            ),
            cancellation_event=cancellation_signal,
        )
    except OSError as exc:
        raise RuntimeError(
            "bounded root-critical kernel worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "root-critical profile cancelled during the kernel worker"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "root-critical profile deadline expired during the kernel worker"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded root-critical kernel worker did not establish a profile"
        )
    return completed.stdout
