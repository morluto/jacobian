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


def run_point_construction_worker(
    request: ProjectiveSingularityPointWorkerRequest,
    *,
    deadline: float,
) -> ProjectiveSingularityPointWorkerComplete:
    """Run one deadline-bound exact chart-to-residue-field transaction."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
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
            completed = run_bounded_process(
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
            )
    except OperationExecutionTimeoutError:
        raise
    except OSError as exc:
        raise RuntimeError(
            "bounded projective singular-point worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during projective singular-point construction"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during projective singular-point construction"
        )
    if completed.stdout_exceeded:
        raise PointConstructionLimitError(
            "projective singular-point result exceeded its exact-output bound"
        )
    if completed.stderr_exceeded or completed.returncode != 0:
        raise RuntimeError(
            "bounded projective singular-point worker did not establish a complete result"
        )
    try:
        return ProjectiveSingularityPointWorkerComplete.model_validate_json(
            completed.stdout,
            strict=True,
        )
    except ValidationError as exc:
        raise RuntimeError(
            "bounded projective singular-point worker returned malformed output"
        ) from exc


__all__ = ["PointConstructionLimitError", "run_point_construction_worker"]
