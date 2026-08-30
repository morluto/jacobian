"""Killable one-shot process boundary for exact field embeddings."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.number_theory.number_fields._embedding_protocol import (
    NUMBER_FIELD_EMBEDDING_WORKER_RESPONSE_ADAPTER,
    NumberFieldEmbeddingWorkerRequest,
    NumberFieldEmbeddingWorkerResponse,
)

_EMBEDDINGS_WORKER = Path(__file__).resolve().with_name("_embeddings_worker.py")
EMBEDDINGS_WORKER_WALL_SECONDS = 120.0
_EMBEDDINGS_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_EMBEDDINGS_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_EMBEDDINGS_WORKER_STDERR_BYTES = 64 * 1024


def run_embeddings_worker(
    request: NumberFieldEmbeddingWorkerRequest,
    *,
    timeout_seconds: float,
    stdout_limit: int,
) -> NumberFieldEmbeddingWorkerResponse:
    """Run one isolated recognition, ordering, and isolation computation."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    try:
        with TemporaryDirectory(
            prefix="jacobian-number-field-embeddings-"
        ) as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_EMBEDDINGS_WORKER)],
                input_bytes=request.model_dump_json().encode("utf-8"),
                timeout_seconds=timeout_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_EMBEDDINGS_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(EMBEDDINGS_WORKER_WALL_SECONDS)),
                    address_space_bytes=_EMBEDDINGS_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_EMBEDDINGS_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded number-field embedding worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during number-field embedding computation"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during number-field embedding computation"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded number-field embedding worker did not establish a profile"
        )

    try:
        return NUMBER_FIELD_EMBEDDING_WORKER_RESPONSE_ADAPTER.validate_json(
            completed.stdout,
            strict=True,
        )
    except ValidationError as exc:
        raise RuntimeError(
            "bounded number-field embedding worker returned malformed output"
        ) from exc


def embeddings_worker_cancelled() -> bool:
    """Report cancellation through the shared bounded-process context."""

    from jacobian.process import bounded_process_cancelled

    return bounded_process_cancelled()


__all__ = [
    "EMBEDDINGS_WORKER_WALL_SECONDS",
    "embeddings_worker_cancelled",
    "run_embeddings_worker",
]
