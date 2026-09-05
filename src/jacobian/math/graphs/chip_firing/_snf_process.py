"""Killable exact normal-form boundary for chip-firing critical groups."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationResourceAdmissionError

_SNF_WORKER = Path(__file__).resolve().with_name("_snf_worker.py")
# Admitted reduced Laplacians satisfy dimension**3 <= 1_500_000. FLINT's
# exact integer SNF on that envelope finishes well under a minute; keep a
# generous killable ceiling so one stalled backend cannot outlive the request.
_SNF_WALL_SECONDS = 120.0
# HNF normalization, residue reduction, transformation admission, worker and
# decoding share this one ceiling. It is a killable safety limit, not the
# mathematical work bound (which lives in _smith_bounds).
_COORDINATE_WALL_SECONDS = 600.0
_SNF_STDERR_LIMIT = 64 * 1024
_SNF_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_SNF_FILE_SIZE_BYTES = 1024 * 1024
_SNF_DIAGNOSTIC_CHARS = 1024


def smith_normal_form_diagonal(matrix: list[list[int]]) -> tuple[int, ...]:
    """Return the SNF diagonal through a deadline-bound killable FLINT worker."""
    return _smith_projection(matrix, None)[0]


def smith_coordinates(
    matrix: list[list[int]], divisor: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute the full exact quotient map inside one killable worker."""
    deadline = _bind_deadline(_COORDINATE_WALL_SECONDS)
    return _smith_projection(matrix, divisor, deadline=deadline)


def _bind_deadline(wall_seconds: float) -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + wall_seconds
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before reduced-Laplacian Smith projection")
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired before Smith projection"
        )
    return deadline


def _smith_projection(
    matrix: list[list[int]], divisor: list[int] | None, *, deadline: float | None = None
) -> tuple[tuple[int, ...], tuple[int, ...]]:

    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    if rows == 0 or cols == 0:
        request_checkpoint("after empty chip-firing Smith projection")
        return (), ()

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    if deadline is None:
        deadline = _bind_deadline(_SNF_WALL_SECONDS)
    request_checkpoint("before reduced-Laplacian Smith projection")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "request deadline expired before reduced-Laplacian SNF"
        )

    request = {
        "matrix": [[format_canonical_integer(value) for value in row] for row in matrix]
    }
    payload = encode_strict_json(
        request
        if divisor is None
        else {
            **request,
            "divisor": [format_canonical_integer(value) for value in divisor],
            "deadline": repr(deadline),
        }
    )
    maximum_entry_digits = max(
        (
            len(format_canonical_integer(value).lstrip("-"))
            for row in matrix
            for value in row
        ),
        default=1,
    )
    maximum_diagonal_digits = max(
        1, min(rows, cols) * (maximum_entry_digits + len(str(max(rows, cols))))
    )
    stdout_limit = len(
        encode_strict_json(
            {
                "diagonal": ["-" + "9" * maximum_diagonal_digits] * min(rows, cols),
                "request_digest": "0" * 64,
            },
        )
    )
    if divisor is not None:
        # Every residue is smaller than a Smith factor. Hadamard bounds
        # their product (the determinant), hence each factor and residue.
        stdout_limit += 32 + min(rows, cols) * (maximum_diagonal_digits + 4)
        stdout_limit = max(stdout_limit, _SNF_DIAGNOSTIC_CHARS + 128)
    try:
        with TemporaryDirectory(prefix="jacobian-chip-firing-snf-") as worker_directory:
            request_checkpoint("before launching reduced-Laplacian Smith worker")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before Smith worker launch"
                )
            completed = run_bounded_process(
                [sys.executable, str(_SNF_WORKER)],
                input_bytes=payload,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_SNF_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_SNF_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_SNF_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded chip-firing SNF worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during reduced-Laplacian SNF"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during reduced-Laplacian SNF"
        )
    request_checkpoint("after reduced-Laplacian Smith worker")
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired after Smith worker"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded chip-firing SNF worker did not establish a diagonal"
        )

    result = _decode_projection(
        completed.stdout, payload, min(rows, cols), divisor is not None
    )
    request_checkpoint("after decoding chip-firing Smith projection")
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired after Smith projection"
        )
    return result


def _decode_projection(
    content: bytes, payload: bytes, dimension: int, has_coordinates: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        response = loads_strict_json(content)
        if (
            has_coordinates
            and isinstance(response, dict)
            and set(response) == {"request_digest", "resource_error"}
        ):
            diagnostic = response["resource_error"]
            if (
                response["request_digest"] != hashlib.sha256(payload).hexdigest()
                or not isinstance(diagnostic, str)
                or len(diagnostic) > _SNF_DIAGNOSTIC_CHARS
            ):
                raise ValueError("worker admission diagnostic is malformed")
            raise OperationResourceAdmissionError(
                location=("graph",),
                code="chip_firing.smith_transform_bound",
                message=diagnostic,
            )
        fields = {
            "diagonal",
            "request_digest",
        }
        if has_coordinates:
            fields.add("coordinates")
        if not isinstance(response, dict) or set(response) != fields:
            raise ValueError("worker response has invalid fields")
        if response["request_digest"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("worker response is not bound to its request")
        diagonal = response["diagonal"]
        if not isinstance(diagonal, list) or len(diagonal) != dimension:
            raise ValueError("worker diagonal is malformed")
        factors = tuple(parse_canonical_integer(value) for value in diagonal)
        if any(d <= 0 for d in factors) or any(b % a for a, b in pairwise(factors)):
            raise ValueError(
                "worker factors must be positive and divide their successors"
            )
        coordinates: tuple[int, ...] = ()
        if has_coordinates:
            values = response["coordinates"]
            nonunit = tuple(d for d in factors if d > 1)
            if not isinstance(values, list) or len(values) != len(nonunit):
                raise ValueError("worker coordinates are malformed")
            coordinates = tuple(parse_canonical_integer(v) for v in values)
            if any(not 0 <= c < d for c, d in zip(coordinates, nonunit, strict=True)):
                raise ValueError("worker coordinate is outside its factor")
        return factors, coordinates
    except OperationResourceAdmissionError:
        raise
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded chip-firing SNF worker returned malformed output"
        ) from exc


__all__ = ["smith_coordinates", "smith_normal_form_diagonal"]
