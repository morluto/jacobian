"""Bounded process owner for exact algebraic inertia."""

from __future__ import annotations

import hashlib
import sys
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.math.matrices._number_field import (
    RecognizedRealSimpleNumberField,
    field_element_coordinates,
)

_INERTIA_WORKER = Path(__file__).with_name("_inertia_worker.py")
_INERTIA_STDOUT_LIMIT = len(
    encode_strict_json(
        {
            "request_digest": "0" * 64,
            "n_positive": 1_000_000,
            "n_negative": 1_000_000,
            "n_zero": 1_000_000,
        },
    )
)
_INERTIA_STDERR_LIMIT = 64 * 1024


def _require_active(deadline: float, phase: str) -> None:
    request_checkpoint(phase)
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def algebraic_inertia_killable(
    matrix: list[list[Any]],
    recognized: RecognizedRealSimpleNumberField,
    *,
    regime: str,
    deadline: float,
) -> tuple[int, int, int]:
    """Compute exact embedded inertia in one deadline-bound child process."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _require_active(deadline, "before exact algebraic inertia")
    payload = encode_strict_json(
        {
            "presentation": recognized.embedding.presentation.model_dump(mode="json"),
            "real_root_index": recognized.embedding.root.real_root_index,
            "regime": regime,
            "matrix": [
                [
                    [
                        f"{coordinate.numerator}/{coordinate.denominator}"
                        for coordinate in reversed(
                            field_element_coordinates(value, recognized)
                        )
                    ]
                    for value in row
                ]
                for row in matrix
            ],
        }
    )
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before exact algebraic inertia"
        )
    with TemporaryDirectory(prefix="jacobian-algebraic-inertia-") as directory:
        completed = run_bounded_process(
            [sys.executable, str(_INERTIA_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=_INERTIA_STDOUT_LIMIT,
            stderr_limit=_INERTIA_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(remaining)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during exact algebraic inertia"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during exact algebraic inertia"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded algebraic inertia worker did not return a result")
    try:
        response = loads_strict_json(completed.stdout)
        if response["request_digest"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("worker request digest mismatch")
        n_positive = response["n_positive"]
        n_negative = response["n_negative"]
        n_zero = response["n_zero"]
        counts = (n_positive, n_negative, n_zero)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("worker inertia counts must be nonnegative integers")
        if sum(counts) != len(matrix):
            raise ValueError("worker inertia counts do not match the matrix order")
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded algebraic inertia worker returned malformed data"
        ) from exc
    _require_active(deadline, "after exact algebraic inertia")
    return n_positive, n_negative, n_zero


__all__ = ["algebraic_inertia_killable"]
