"""Killable process boundary for scaled radix-prefix root isolation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
    run_bounded_process,
    worker_environment,
)

_WORKER = Path(__file__).resolve().with_name("_radix_prefix_worker.py")
RADIX_ISOLATION_OWNER_SECONDS = 600.0
_WORKER_WALL_SECONDS = RADIX_ISOLATION_OWNER_SECONDS
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDOUT_BYTES = 256 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


def run_scaled_integer_part_worker(
    *,
    polynomial: tuple[int, ...],
    real_root_index: int,
    scale: int,
    isolation_bits: int,
    deadline: float,
) -> int:
    """Isolate one scaled algebraic integer part in a killable child."""

    timeout_seconds = deadline - time.monotonic()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before radix isolation worker launch"
        )
    payload = json.dumps(
        {
            "polynomial": [str(int(coefficient)) for coefficient in polynomial],
            "real_root_index": real_root_index,
            "scale": str(int(scale)),
            "isolation_bits": isolation_bits,
        }
    ).encode("utf-8")
    try:
        with TemporaryDirectory(prefix="jacobian-radix-prefix-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=payload,
                timeout_seconds=timeout_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_STDOUT_BYTES,
                stderr_limit=_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(_WORKER_WALL_SECONDS)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    check_bounded_process_result(completed)
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc
    if not response.get("ok"):
        code = response.get("code")
        message = str(response.get("message", "radix isolation failed"))
        if code == "root_index":
            raise OperationDomainValidationError(
                location=("value", "real_root_index"),
                code="algebraic_number.radix_root_index",
                message=message,
            )
        if code == "not_irreducible":
            raise OperationDomainValidationError(
                location=("value",),
                code="real_algebraic.not_irreducible",
                message=message,
            )
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return int(parse_canonical_integer(response["scaled_floor"]))


__all__ = ["RADIX_ISOLATION_OWNER_SECONDS", "run_scaled_integer_part_worker"]
