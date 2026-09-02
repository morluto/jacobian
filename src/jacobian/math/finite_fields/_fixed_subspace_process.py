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


def _require_active(deadline: float, stage: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"finite-field fixed-subspace computation cancelled {stage}"
        )
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"finite-field fixed-subspace deadline expired {stage}"
        )


def _run_fixed_subspace_worker(
    input_bytes: bytes,
    *,
    deadline: float,
    stdout_limit: int,
) -> bytes:
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _require_active(deadline, "before linear algebra")
    remaining = deadline - time.monotonic()
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
            "bounded finite-field fixed-subspace worker did not establish a result"
        )
    return completed.stdout


def _matrix_worker_input(matrix: PrimeFieldMatrix, *, operation: str) -> bytes:
    return json.dumps(
        {
            "operation": operation,
            "prime": matrix.prime,
            "entries": [list(row) for row in matrix.entries],
            "columns": matrix.columns,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def run_fixed_subspace_linear_algebra(
    equation_matrix: PrimeFieldMatrix,
    *,
    deadline: float,
) -> tuple[tuple[int, ...], ...]:
    """Run backend nullspace/RREF work behind a killable process boundary."""

    input_bytes = _matrix_worker_input(equation_matrix, operation="basis")
    monomial_count = equation_matrix.columns
    scalar_bytes = max(1, len(str(equation_matrix.prime - 1)))
    stdout_limit = max(
        4_096,
        128 + monomial_count * monomial_count * (scalar_bytes + 4),
    )
    stdout = _run_fixed_subspace_worker(
        input_bytes,
        deadline=deadline,
        stdout_limit=stdout_limit,
    )
    try:
        decoded = json.loads(stdout.decode("utf-8"))
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


def run_fixed_subspace_generator_validation(
    generator_matrices: tuple[PrimeFieldMatrix, ...],
    *,
    deadline: float,
) -> bool:
    """Check generator invertibility behind the same killable worker boundary."""

    variable_count = generator_matrices[0].columns
    input_bytes = _generator_worker_input(generator_matrices, deadline=deadline)
    stdout = _run_fixed_subspace_worker(
        input_bytes,
        deadline=deadline,
        stdout_limit=_generator_ranks_stdout_limit(
            len(generator_matrices), variable_count
        ),
    )
    try:
        decoded = json.loads(stdout.decode("utf-8"))
        ranks = decoded["ranks"]
        if not isinstance(ranks, list) or len(ranks) != len(generator_matrices):
            raise ValueError("malformed generator ranks")
        if any(type(generator_rank) is not int for generator_rank in ranks):
            raise ValueError("malformed generator rank")
        return all(generator_rank == variable_count for generator_rank in ranks)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "bounded finite-field fixed-subspace worker returned malformed rank"
        ) from exc


def _generator_worker_input(
    generator_matrices: tuple[PrimeFieldMatrix, ...], *, deadline: float
) -> bytes:
    """Marshal admitted generators while observing the request envelope."""

    first = generator_matrices[0]
    encoded = bytearray(
        f'{{"operation":"generator_ranks","prime":{first.prime},'
        f'"columns":{first.columns},"matrices":['.encode()
    )
    for matrix_index, generator in enumerate(generator_matrices):
        _require_active(deadline, "during generator marshalling")
        if matrix_index:
            encoded.append(ord(","))
        encoded.append(ord("["))
        for row_index, row in enumerate(generator.entries):
            if row_index:
                encoded.append(ord(","))
            encoded.extend(json.dumps(row, separators=(",", ":")).encode("utf-8"))
            _require_active(deadline, "during generator marshalling")
        encoded.append(ord("]"))
    encoded.extend(b"]}")
    return bytes(encoded)


def _generator_ranks_stdout_limit(generator_count: int, variable_count: int) -> int:
    """Return the exact largest compact JSON rank-projection size."""

    rank_digits = len(str(variable_count))
    return 11 + generator_count * (rank_digits + 1)


__all__ = [
    "run_fixed_subspace_generator_validation",
    "run_fixed_subspace_linear_algebra",
]
