"""Bounded process owner for Riemann-form Smith reduction."""

from __future__ import annotations

import hashlib
import json
import sys
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.matrices.values import IntegerMatrix, SmithNormalForm

_SMITH_WORKER = Path(__file__).with_name("_smith_worker.py")
_SMITH_STDERR_LIMIT = 64 * 1024


def _require_active(deadline: float, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"complex-torus deadline expired {phase}")


def smith_normal_form_killable(
    matrix: IntegerMatrix,
    *,
    deadline: float,
) -> SmithNormalForm:
    """Compute Riemann-form Smith data in a deadline-bound child process."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _require_active(deadline, "before alternating Smith kernel")
    payload = json.dumps(
        {
            "entries": [
                [
                    format_canonical_integer(parse_canonical_integer(value))
                    for value in row
                ]
                for row in matrix.entries
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "complex-torus deadline expired before alternating Smith kernel"
        )
    with TemporaryDirectory(prefix="jacobian-riemann-smith-") as directory:
        completed = run_bounded_process(
            [sys.executable, str(_SMITH_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=CanonicalLimits().max_output_bytes,
            stderr_limit=_SMITH_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(remaining)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "complex-torus request cancelled during alternating Smith kernel"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "complex-torus deadline expired during alternating Smith kernel"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded alternating Smith worker did not return a form")
    try:
        response = loads_strict_json(completed.stdout)
        if response["request_digest"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("worker request digest mismatch")
        raw_normal_form = response["normal_form"]
        if not isinstance(raw_normal_form, list):
            raise ValueError("worker Smith form must be a list")
        normal_form = []
        for row in raw_normal_form:
            if not isinstance(row, list):
                raise ValueError("worker Smith form rows must be lists")
            decoded_row = []
            for value in row:
                if not isinstance(value, str):
                    raise ValueError("worker integer must be a canonical string")
                decoded = parse_canonical_integer(value)
                if format_canonical_integer(decoded) != value:
                    raise ValueError("worker integer is not canonical")
                decoded_row.append(decoded)
            normal_form.append(decoded_row)
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded alternating Smith worker returned malformed data"
        ) from exc
    dimension = len(matrix.entries)
    if len(normal_form) != dimension or any(
        len(row) != dimension for row in normal_form
    ):
        raise RuntimeError(
            "bounded alternating Smith worker returned invalid dimensions"
        )
    diagonal = tuple(normal_form[index][index] for index in range(dimension))
    rank = next(
        (index for index, value in enumerate(diagonal) if value == 0), dimension
    )
    if any(value == 0 for value in diagonal[:rank]) or any(
        value != 0 for value in diagonal[rank:]
    ):
        raise RuntimeError(
            "bounded alternating Smith worker returned an invalid diagonal"
        )
    factors = tuple(abs(value) for value in diagonal[:rank])
    canonical_normal_form = IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(factors[row])
                if row == column and row < rank
                else "0"
                for column in range(dimension)
            )
            for row in range(dimension)
        )
    )
    _require_active(deadline, "after alternating Smith kernel")
    return SmithNormalForm(
        normal_form=canonical_normal_form,
        rank=rank,
        invariant_factors=tuple(format_canonical_integer(value) for value in factors),
    )


__all__ = ["smith_normal_form_killable"]
