"""Killable SymPy Smith process boundary for integral chain homology."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancelled,
)
from jacobian.math.matrices.certified_snf.operations import Matrix, SmithReduction

_SMITH_WORKER = Path(__file__).resolve().with_name("_smith_worker.py")
_SMITH_STDERR_LIMIT = 64 * 1024
_SMITH_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_SMITH_FILE_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SmithProcessResult:
    """Strictly decoded derived projection from one trusted worker."""

    reduction: SmithReduction
    left_inverse: Matrix
    right_inverse: Matrix


def _require_active_deadline(deadline: float, *, stage: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"request cancelled {stage} integral-homology Smith reduction"
        )
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"integral homology deadline expired {stage} Smith worker"
        )


def _positive_remaining_allowance(deadline: float, *, stage: str) -> float:
    """Return one positive launch allowance or a typed execution outcome."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"request cancelled {stage} integral-homology Smith reduction"
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            f"integral homology deadline expired {stage} Smith worker"
        )
    return remaining


def _decimal_digits_for_bits(bits: int) -> int:
    return max(1, (max(1, bits) * 30_103 + 99_999) // 100_000 + 1)


def _stdout_limit(
    *,
    rows: int,
    columns: int,
    left_bits: int,
    right_bits: int,
    diagonal_bits: int,
    left_inverse_bits: int,
    right_inverse_bits: int,
) -> int:
    """Bound the concrete JSON worker projection before process launch."""

    weighted_scalars = (
        rows * columns * (_decimal_digits_for_bits(diagonal_bits) + 4)
        + rows * rows * (_decimal_digits_for_bits(left_bits) + 4)
        + columns * columns * (_decimal_digits_for_bits(right_bits) + 4)
        + rows * rows * (_decimal_digits_for_bits(left_inverse_bits) + 4)
        + columns * columns * (_decimal_digits_for_bits(right_inverse_bits) + 4)
    )
    # Object/array punctuation, factors, determinant fields, and digest.
    return 4096 + weighted_scalars + 16 * (rows + columns + 1)


def _strict_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Smith worker scalar is not an integer")
    return value


def _strict_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    maximum_bits: int,
) -> Matrix:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("Smith worker matrix row count is invalid")
    result: Matrix = []
    for candidate_row in value:
        if not isinstance(candidate_row, list) or len(candidate_row) != columns:
            raise ValueError("Smith worker matrix column count is invalid")
        row = [_strict_int(item) for item in candidate_row]
        if any(abs(item).bit_length() > maximum_bits for item in row):
            raise ValueError("Smith worker matrix exceeded its admitted height")
        result.append(row)
    return result


def _decode_result(
    payload: Any,
    *,
    request_digest: str,
    source: Matrix,
    rows: int,
    columns: int,
    left_bits: int,
    right_bits: int,
    diagonal_bits: int,
    left_inverse_bits: int,
    right_inverse_bits: int,
) -> SmithProcessResult:
    if not isinstance(payload, dict) or set(payload) != {
        "request_digest",
        "diagonal",
        "left",
        "right",
        "rank",
        "invariant_factors",
        "left_determinant",
        "right_determinant",
        "left_inverse",
        "right_inverse",
    }:
        raise ValueError("Smith worker result shape is invalid")
    if payload["request_digest"] != request_digest:
        raise ValueError("Smith worker result is not bound to the admitted request")
    diagonal = _strict_matrix(
        payload["diagonal"],
        rows=rows,
        columns=columns,
        maximum_bits=diagonal_bits,
    )
    left = _strict_matrix(
        payload["left"], rows=rows, columns=rows, maximum_bits=left_bits
    )
    right = _strict_matrix(
        payload["right"],
        rows=columns,
        columns=columns,
        maximum_bits=right_bits,
    )
    left_inverse = _strict_matrix(
        payload["left_inverse"],
        rows=rows,
        columns=rows,
        maximum_bits=left_inverse_bits,
    )
    right_inverse = _strict_matrix(
        payload["right_inverse"],
        rows=columns,
        columns=columns,
        maximum_bits=right_inverse_bits,
    )
    rank = _strict_int(payload["rank"])
    factors_value = payload["invariant_factors"]
    if not isinstance(factors_value, list):
        raise ValueError("Smith worker factors are invalid")
    factors = tuple(_strict_int(value) for value in factors_value)
    diagonal_values = tuple(
        diagonal[index][index] for index in range(min(rows, columns))
    )
    if (
        rank != len(factors)
        or rank > min(rows, columns)
        or any(value <= 0 for value in factors)
        or diagonal_values[:rank] != factors
        or any(value != 0 for value in diagonal_values[rank:])
        or any(
            right_factor % left_factor
            for left_factor, right_factor in pairwise(factors)
        )
        or any(
            diagonal[row][column] != 0
            for row in range(rows)
            for column in range(columns)
            if row != column
        )
    ):
        raise ValueError("Smith worker diagonal is not canonical")
    left_determinant = _strict_int(payload["left_determinant"])
    right_determinant = _strict_int(payload["right_determinant"])
    if abs(left_determinant) != 1 or abs(right_determinant) != 1:
        raise ValueError("Smith worker transformation determinant is invalid")
    return SmithProcessResult(
        reduction=SmithReduction(
            source=[row[:] for row in source],
            diagonal=diagonal,
            left=left,
            right=right,
            rank=rank,
            invariant_factors=factors,
            left_determinant=left_determinant,
            right_determinant=right_determinant,
        ),
        left_inverse=left_inverse,
        right_inverse=right_inverse,
    )


def smith_reduce_in_worker(
    source: Matrix,
    *,
    rows: int,
    columns: int,
    deadline: float,
    left_bits: int,
    right_bits: int,
    diagonal_bits: int,
    left_inverse_bits: int,
    right_inverse_bits: int,
) -> SmithProcessResult:
    """Run one admitted exact Smith decomposition in a killable child."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    input_bytes = json.dumps(
        {"matrix": source, "row_count": rows, "column_count": columns},
        separators=(",", ":"),
    ).encode("utf-8")
    request_digest = hashlib.sha256(input_bytes).hexdigest()
    stdout_limit = _stdout_limit(
        rows=rows,
        columns=columns,
        left_bits=left_bits,
        right_bits=right_bits,
        diagonal_bits=diagonal_bits,
        left_inverse_bits=left_inverse_bits,
        right_inverse_bits=right_inverse_bits,
    )
    _require_active_deadline(deadline, stage="before")
    try:
        with TemporaryDirectory(
            prefix="jacobian-integral-homology-smith-"
        ) as directory:
            # Temporary-directory setup belongs to the inherited request
            # envelope. Recompute the allowance at the actual launch boundary
            # so setup cannot mint a fresh child-process clock.
            remaining = _positive_remaining_allowance(
                deadline,
                stage="before launching the",
            )
            completed = run_bounded_process(
                [sys.executable, str(_SMITH_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_SMITH_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_SMITH_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_SMITH_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError(
            "bounded integral-homology Smith worker could not start"
        ) from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during integral-homology Smith reduction"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "integral homology deadline expired during Smith reduction"
        )
    _require_active_deadline(deadline, stage="after cleaning up the")
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded integral-homology Smith worker did not establish a reduction"
        )
    try:
        decoded = json.loads(completed.stdout.decode("utf-8"))
        result = _decode_result(
            decoded,
            request_digest=request_digest,
            source=source,
            rows=rows,
            columns=columns,
            left_bits=left_bits,
            right_bits=right_bits,
            diagonal_bits=diagonal_bits,
            left_inverse_bits=left_inverse_bits,
            right_inverse_bits=right_inverse_bits,
        )
        _require_active_deadline(deadline, stage="after decoding the")
        return result
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded integral-homology Smith worker returned malformed output"
        ) from exc


__all__ = ["SmithProcessResult", "smith_reduce_in_worker"]
