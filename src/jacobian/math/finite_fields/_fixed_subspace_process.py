"""Bounded process owner for finite-field fixed-subspace linear algebra."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancelled,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

_FIXED_SUBSPACE_WORKER = Path(__file__).with_name("_fixed_subspace_worker.py")
_FIXED_SUBSPACE_STDERR_BYTES = 64 * 1024
_FIXED_SUBSPACE_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_FIXED_SUBSPACE_FILE_SIZE_BYTES = 1024 * 1024


def run_fixed_subspace_linear_algebra(
    equation_matrix: PrimeFieldMatrix,
    *,
    deadline: float,
) -> tuple[tuple[int, ...], ...]:
    """Run backend nullspace/RREF work behind a killable process boundary."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    remaining = deadline - time.monotonic()
    if request_cancelled():
        raise OperationExecutionCancelledError(
            "finite-field fixed-subspace computation cancelled before linear algebra"
        )
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "finite-field fixed-subspace deadline expired before linear algebra"
        )
    input_bytes = json.dumps(
        {
            "prime": equation_matrix.prime,
            "entries": [list(row) for row in equation_matrix.entries],
            "columns": equation_matrix.columns,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    monomial_count = equation_matrix.columns
    scalar_bytes = max(1, len(str(equation_matrix.prime - 1)))
    stdout_limit = max(
        4_096,
        128 + monomial_count * monomial_count * (scalar_bytes + 4),
    )
    try:
        with TemporaryDirectory(prefix="jacobian-fixed-subspace-") as directory:
            completed = run_bounded_process(
                [sys.executable, str(_FIXED_SUBSPACE_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_FIXED_SUBSPACE_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_FIXED_SUBSPACE_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FIXED_SUBSPACE_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded finite-field fixed-subspace worker could not start"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "finite-field fixed-subspace computation cancelled during linear algebra"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "finite-field fixed-subspace deadline expired during linear algebra"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded finite-field fixed-subspace worker did not establish a basis"
        )
    try:
        decoded = json.loads(completed.stdout.decode("utf-8"))
        raw_basis = decoded["basis_rows"]
        if not isinstance(raw_basis, list) or len(raw_basis) > monomial_count:
            raise ValueError("malformed fixed-subspace basis row count")
        basis_rows: list[tuple[int, ...]] = []
        for raw_row in raw_basis:
            if not isinstance(raw_row, list) or len(raw_row) != monomial_count:
                raise ValueError("malformed fixed-subspace basis row")
            if any(type(value) is not int for value in raw_row):
                raise ValueError("malformed fixed-subspace basis coefficient")
            if any(not 0 <= value < equation_matrix.prime for value in raw_row):
                raise ValueError("fixed-subspace basis coefficient is out of range")
            basis_rows.append(tuple(raw_row))
        return tuple(basis_rows)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "bounded finite-field fixed-subspace worker returned malformed basis"
        ) from exc


__all__ = ["run_fixed_subspace_linear_algebra"]
