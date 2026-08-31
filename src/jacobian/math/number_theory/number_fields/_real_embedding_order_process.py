"""Killable one-shot process boundary for selected-image isolation."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.number_theory.number_fields._real_embedding_order_protocol import (
    SelectedImageWorkerRequest,
)

_WORKER = Path(__file__).resolve().with_name("_real_embedding_order_worker.py")
_WORKER_WALL_SECONDS = 600.0
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDOUT_BYTES = 128 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


def run_selected_image_worker(
    request: SelectedImageWorkerRequest,
    *,
    deadline: float,
) -> bytes:
    """Run one isolated selected-image isolation computation."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    timeout_seconds = deadline - time.monotonic()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before selected-image worker launch"
        )

    try:
        with TemporaryDirectory(prefix="jacobian-selected-image-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=request.model_dump_json().encode("utf-8"),
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
        raise RuntimeError(
            "bounded selected-image worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during selected-image isolation"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during selected-image isolation"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded selected-image worker did not establish a result")

    return completed.stdout


__all__ = [
    "run_selected_image_worker",
]
