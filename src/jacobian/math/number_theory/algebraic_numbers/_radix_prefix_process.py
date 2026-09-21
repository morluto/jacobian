"""Killable process boundary for scaled radix-prefix root isolation."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_WORKER = Path(__file__).resolve().with_name("_radix_prefix_worker.py")
RADIX_ISOLATION_OWNER_SECONDS = 600.0
_WORKER_WALL_SECONDS = RADIX_ISOLATION_OWNER_SECONDS
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDOUT_BYTES = 256 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class RadixIsolationSuccess:
    scaled_floor: int


@dataclass(frozen=True, slots=True)
class RadixIsolationDomainError:
    code: Literal["root_index", "not_irreducible"]
    message: str


@dataclass(frozen=True, slots=True)
class RadixIsolationFailure:
    kind: Literal["refinement"]
    message: str


def _decode_radix_result(
    value: object, *, scaled_floor_digit_bound: int
) -> RadixIsolationSuccess | RadixIsolationDomainError | RadixIsolationFailure:
    if not isinstance(value, dict):
        raise ValueError("radix worker result must be an object")
    tag = value.get("tag")
    if tag == "success":
        if set(value) != {"tag", "scaled_floor"}:
            raise ValueError("malformed radix success")
        scaled_floor = value.get("scaled_floor")
        if (
            not isinstance(scaled_floor, str)
            or len(scaled_floor) > scaled_floor_digit_bound
        ):
            raise ValueError("malformed radix floor")
        try:
            parsed = int(parse_canonical_integer(scaled_floor))
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed radix floor") from exc
        return RadixIsolationSuccess(parsed)
    if tag == "domain_error":
        if set(value) != {"tag", "code", "message"}:
            raise ValueError("malformed radix domain error")
        code, message = value.get("code"), value.get("message")
        if code not in ("root_index", "not_irreducible") or not isinstance(
            message, str
        ):
            raise ValueError("malformed radix domain error")
        return RadixIsolationDomainError(code, message)
    if tag == "refinement":
        if set(value) != {"tag", "message"} or not isinstance(
            value.get("message"), str
        ):
            raise ValueError("malformed radix refinement")
        return RadixIsolationFailure("refinement", value["message"])
    raise ValueError("unknown radix worker result")


def run_scaled_integer_part_worker(
    *,
    polynomial: tuple[int, ...],
    real_root_index: int,
    scale: int,
    isolation_bits: int,
    deadline: float,
    scaled_floor_digit_bound: int,
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
            outcome = run_checked_worker_process(
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
                max_frame_bytes=_WORKER_STDOUT_BYTES,
                decode_result=lambda value: _decode_radix_result(
                    value, scaled_floor_digit_bound=scaled_floor_digit_bound
                ),
            )
    except OSError as exc:
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    if isinstance(outcome, RadixIsolationSuccess):
        return outcome.scaled_floor
    if isinstance(outcome, RadixIsolationDomainError):
        if outcome.code == "root_index":
            raise OperationDomainValidationError(
                location=("value", "real_root_index"),
                code="algebraic_number.radix_root_index",
                message=outcome.message,
            )
        raise OperationDomainValidationError(
            location=("value",),
            code="real_algebraic.not_irreducible",
            message=outcome.message,
        )
    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)


__all__ = ["RADIX_ISOLATION_OWNER_SECONDS", "run_scaled_integer_part_worker"]
