"""Killable unit-disk signature, factorization, and real-gcd kernel."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_root_profile_worker.py")
_STDOUT_BYTES = 64 * 1024
_STDERR_BYTES = 64 * 1024
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0


def _decode_root_profile_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("unit-disk worker result must be an object")
    return value


def count_reduced_roots(
    coefficients: list[int], *, deadline: float
) -> tuple[int, int, int]:
    """Count inside/on/outside roots of one admitted primitive integer polynomial."""

    remaining = deadline - monotonic() - _PARENT_FINALIZATION_SECONDS
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "unit-disk root profile deadline expired before the kernel worker"
        )
    payload = encode_strict_json(
        {
            "coefficients": [
                format_canonical_integer(coefficient) for coefficient in coefficients
            ]
        }
    )
    try:
        with TemporaryDirectory(prefix="jacobian-unit-disk-") as worker_directory:
            response = run_checked_worker_process(
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
                decode_result=_decode_root_profile_result,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(
            "bounded unit-disk kernel worker could not be started"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "ok"
        or any(
            type(response.get(key)) is not int for key in ("inside", "on", "outside")
        )
    ):
        raise RuntimeError("bounded unit-disk kernel worker returned malformed output")
    inside = response["inside"]
    on = response["on"]
    outside = response["outside"]
    if min(inside, on, outside) < 0:
        raise RuntimeError("bounded unit-disk kernel worker returned malformed output")
    if inside + on + outside != len(coefficients) - 1:
        raise RuntimeError(
            "bounded unit-disk kernel worker returned counts that do not match the admitted degree"
        )
    return inside, on, outside
