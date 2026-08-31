"""Bounded process owner for exact algebraic sign isolation."""

from __future__ import annotations

import hashlib
import json
import sys
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancelled,
)
from jacobian.canonical import CanonicalizationError, loads_strict_json
from jacobian.math.matrices._number_field import (
    RecognizedRealSimpleNumberField,
    field_element_coordinates,
)

_SIGN_WORKER = Path(__file__).with_name("_inertia_worker.py")
_SIGN_STDOUT_LIMIT = 64 * 1024
_SIGN_STDERR_LIMIT = 64 * 1024


def _require_active(deadline: float, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def field_element_sign_killable(
    value: Any,
    recognized: RecognizedRealSimpleNumberField,
    *,
    deadline: float,
) -> int:
    """Determine one embedded sign in a deadline-bound child process."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _require_active(deadline, "before exact algebraic sign isolation")
    coordinates = field_element_coordinates(value, recognized)
    payload = json.dumps(
        {
            "presentation": recognized.embedding.presentation.model_dump(mode="json"),
            "real_root_index": recognized.embedding.root.real_root_index,
            "coefficients_descending": [
                f"{coordinate.numerator}/{coordinate.denominator}"
                for coordinate in reversed(coordinates)
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before exact algebraic sign isolation"
        )
    with TemporaryDirectory(prefix="jacobian-inertia-sign-") as directory:
        completed = run_bounded_process(
            [sys.executable, str(_SIGN_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=_SIGN_STDOUT_LIMIT,
            stderr_limit=_SIGN_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(remaining)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during exact algebraic sign isolation"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during exact algebraic sign isolation"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded algebraic sign worker did not return a sign")
    try:
        response = loads_strict_json(completed.stdout)
        if response["request_digest"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("worker request digest mismatch")
        sign = response["sign"]
        if not isinstance(sign, int) or sign not in (-1, 0, 1):
            raise ValueError("worker sign must be -1, 0, or 1")
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded algebraic sign worker returned malformed data"
        ) from exc
    _require_active(deadline, "after exact algebraic sign isolation")
    return sign


__all__ = ["field_element_sign_killable"]
