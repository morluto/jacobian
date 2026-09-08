"""Killable unit-disk signature, factorization, and real-gcd kernel."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_root_profile_worker.py")
_STDOUT_BYTES = 64 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0


def count_reduced_roots(
    coefficients: list[int], *, deadline: float
) -> tuple[int, int, int]:
    """Count inside/on/outside roots of one admitted primitive integer polynomial."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "unit-disk root profile deadline expired before the kernel worker"
        )
    payload = encode_strict_json({"coefficients": coefficients})
    try:
        with TemporaryDirectory(prefix="jacobian-unit-disk-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER_PATH)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_BYTES,
                stderr_limit=_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_STDOUT_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded unit-disk kernel worker could not be started"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "unit-disk root profile cancelled during the kernel worker"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "unit-disk root profile deadline expired during the kernel worker"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded unit-disk kernel worker did not return root counts")
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=_STDOUT_BYTES,
                max_output_bytes=_STDOUT_BYTES,
            ),
        )
    except CanonicalizationError as exc:
        raise RuntimeError(
            "bounded unit-disk kernel worker returned malformed output"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or any(type(response.get(key)) is not int for key in ("inside", "on", "outside"))
    ):
        raise RuntimeError("bounded unit-disk kernel worker returned malformed output")
    inside = response["inside"]
    on = response["on"]
    outside = response["outside"]
    if min(inside, on, outside) < 0:
        raise RuntimeError("bounded unit-disk kernel worker returned malformed output")
    return inside, on, outside
