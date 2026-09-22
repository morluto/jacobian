"""Killable process boundary for exact projective singular-point construction."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.algebraic_curves._singularity_point_worker import (
    ProjectiveSingularityPointWorkerComplete,
    ProjectiveSingularityPointWorkerRequest,
)

_POINT_WORKER = Path(__file__).resolve().with_name("_singularity_point_worker.py")
_POINT_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_POINT_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_POINT_WORKER_STDOUT_BYTES = 256 * 1024
_POINT_WORKER_STDERR_BYTES = 64 * 1024


class PointConstructionLimitError(RuntimeError):
    """The isolated point result exceeded its proved exact-output envelope."""


def _decode_point_result(value: object) -> object:
    if not isinstance(value, dict):
        raise ValueError("projective singular-point result must be an object")
    return value


def run_point_construction_worker(
    request: ProjectiveSingularityPointWorkerRequest,
    *,
    deadline: float,
) -> ProjectiveSingularityPointWorkerComplete:
    """Run one deadline-bound exact chart-to-residue-field transaction."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_checked_worker_process,
        worker_environment,
    )

    payload = request.model_dump_json().encode("utf-8")
    try:
        with TemporaryDirectory(
            prefix="jacobian-projective-singular-points-"
        ) as worker_directory:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before projective point construction"
                )
            response = run_checked_worker_process(
                [sys.executable, str(_POINT_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_POINT_WORKER_STDOUT_BYTES,
                stderr_limit=_POINT_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_POINT_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_POINT_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
                decode_result=_decode_point_result,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(
            "bounded projective singular-point worker could not be started"
        ) from exc

    try:
        return ProjectiveSingularityPointWorkerComplete.model_validate_json(
            encode_strict_json(response)
        )
    except ValidationError as exc:
        raise RuntimeError(
            "bounded projective singular-point worker returned malformed output"
        ) from exc


__all__ = ["PointConstructionLimitError", "run_point_construction_worker"]
