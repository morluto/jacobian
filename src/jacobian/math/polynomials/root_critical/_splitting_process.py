"""Bounded child-process owner for exact splitting-field kernels.

The kernel runs in a killable child so a wedged SymPy ``all_roots``,
``to_number_field``, or ``minimal_polynomial`` call cannot hold the request.
Process ownership lives here rather than in the public operations module.
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
    run_checked_worker_process,
    worker_environment,
)

SPLITTING_WORKER_PATH = Path(__file__).resolve().with_name("_splitting_worker.py")
SPLITTING_STDOUT_BYTES = 64 * 1024 * 1024
SPLITTING_STDERR_BYTES = 64 * 1024
SPLITTING_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024

__all__ = ["run_splitting_worker_process"]


def _decode_splitting_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("splitting-field worker result must be an object")
    return value


def run_splitting_worker_process(
    polynomial: RationalPolynomial,
    *,
    mode: str,
    embedding_index: int,
    max_pair_rows: object,
    remaining_seconds: float,
    cancellation_signal: RequestCancellationSignal | None,
) -> dict[str, object]:
    """Run the splitting-field worker once and return its validated stdout bytes."""

    if remaining_seconds <= 0:
        raise OperationExecutionTimeoutError(
            "splitting-field deadline expired before the kernel worker"
        )
    payload = encode_strict_json(
        {
            "mode": mode,
            "polynomial": polynomial.model_dump_json(),
            "embedding_index": embedding_index,
            "max_pair_rows": max_pair_rows,
        }
    )
    try:
        response = run_checked_worker_process(
            [sys.executable, str(SPLITTING_WORKER_PATH)],
            input_bytes=payload,
            timeout_seconds=remaining_seconds,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=SPLITTING_STDOUT_BYTES,
            stderr_limit=SPLITTING_STDERR_BYTES,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(remaining_seconds)),
                address_space_bytes=SPLITTING_ADDRESS_SPACE_BYTES,
                file_size_bytes=SPLITTING_STDOUT_BYTES,
            ),
            cancellation_event=cancellation_signal,
            decode_result=_decode_splitting_result,
        )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(
            "bounded splitting-field kernel worker could not be started"
        ) from exc
    return response
