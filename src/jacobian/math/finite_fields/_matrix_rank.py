"""Finite-field matrix rank operation declaration."""

from __future__ import annotations

from functools import partial
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.finite_fields._flint import (
    context as _backend_context,
)
from jacobian.math.finite_fields._flint import (
    to_backend as _to_backend,
)
from jacobian.math.finite_fields._matrix_rank_kernels import (
    compute_matrix_rank as _compute_matrix_rank_kernel,
)
from jacobian.math.finite_fields._matrix_rank_models import (
    MatrixRankRequest,
    MatrixRankResult,
)
from jacobian.math.finite_fields.values import AxisBoundMatrix

_MATRIX_RANK_WALL_SECONDS = 600.0


def _execution_deadline() -> float:
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + _MATRIX_RANK_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _require_deadline(deadline: float, stage: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"finite-field rank cancelled {stage}")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"finite-field rank deadline expired {stage}"
        )


def _largest_independent_indices(
    vectors: list[tuple[Any, ...]],
    *,
    presentation: Any,
    ordered_indices: list[int],
) -> list[int]:
    """Select a maximum-weight independent subset of finite-field vectors."""

    active_context = _backend_context(presentation)
    zero = active_context.zero()
    basis: list[tuple[int, list[Any]]] = []
    selected: list[int] = []
    for index in ordered_indices:
        candidate = [
            _to_backend(value, active_context=active_context)
            for value in vectors[index]
        ]
        for pivot, row in basis:
            factor = candidate[pivot]
            if factor != zero:
                for column in range(pivot, len(candidate)):
                    candidate[column] -= factor * row[column]
        pivot = next(
            (column for column, value in enumerate(candidate) if value != zero),
            None,
        )
        if pivot is None:
            continue
        pivot_value = candidate[pivot]
        for column in range(pivot, len(candidate)):
            candidate[column] /= pivot_value
        basis.append((pivot, candidate))
        selected.append(index)
    return selected


_FIELD: dict[str, object] = {
    "characteristic": 2,
    "modulus_coefficients": [0, 1],
    "generator": "a",
}

_MATRIX: dict[str, object] = {
    "presentation": _FIELD,
    "row_axis": {"name": "rows", "labels": ["r0", "r1"]},
    "column_axis": {"name": "cols", "labels": ["c0", "c1"]},
    "entries": [
        [
            {"presentation": _FIELD, "coordinates": [1]},
            {"presentation": _FIELD, "coordinates": [1]},
        ],
        [
            {"presentation": _FIELD, "coordinates": [1]},
            {"presentation": _FIELD, "coordinates": [1]},
        ],
    ],
}


def compute_matrix_rank(
    matrix: AxisBoundMatrix,
    *,
    enforce_transport_limit: bool = False,
) -> MatrixRankResult:
    """Return the exact rank of a labelled matrix over its presented finite field."""
    deadline = _execution_deadline()
    execution_checkpoint = partial(_require_deadline, deadline)
    execution_checkpoint("before result admission")
    if enforce_transport_limit:
        # Reserve the complete result envelope before the backend call so
        # that we do not waste CPU on a result known to be undeliverable.
        # The worst case is full rank with all pivot labels present.
        row_vectors = [tuple(row) for row in matrix.entries]
        column_vectors = [
            tuple(matrix.entries[row][index] for row in range(len(matrix.entries)))
            for index in range(len(matrix.column_axis.labels))
        ]
        row_order = sorted(
            range(len(row_vectors)),
            key=lambda index: (
                len(encode_strict_json(matrix.row_axis.labels[index])),
                matrix.row_axis.labels[index],
            ),
            reverse=True,
        )
        column_order = sorted(
            range(len(column_vectors)),
            key=lambda index: (
                len(encode_strict_json(matrix.column_axis.labels[index])),
                matrix.column_axis.labels[index],
            ),
            reverse=True,
        )
        possible_rows = _largest_independent_indices(
            row_vectors,
            presentation=matrix.presentation,
            ordered_indices=row_order,
        )
        possible_columns = _largest_independent_indices(
            column_vectors,
            presentation=matrix.presentation,
            ordered_indices=column_order,
        )
        max_rank = min(len(possible_rows), len(possible_columns))
        try:
            pivot_row_labels = [
                matrix.row_axis.labels[index] for index in possible_rows[:max_rank]
            ]
            pivot_column_labels = [
                matrix.column_axis.labels[index]
                for index in possible_columns[:max_rank]
            ]
            result_probe = encode_strict_json(
                {
                    "matrix": matrix.model_dump(mode="json"),
                    "rank": max_rank,
                    "pivot_rows": pivot_row_labels,
                    "pivot_columns": pivot_column_labels,
                }
            )
        except CanonicalizationError as exc:
            raise OperationDomainValidationError(
                location=("matrix",),
                code="finite_field.matrix_rank.result_bound",
                message="matrix-rank result exceeds the canonical output bound",
            ) from exc
        if len(result_probe) > CanonicalLimits().max_output_bytes:
            raise OperationDomainValidationError(
                location=("matrix",),
                code="finite_field.matrix_rank.result_bound",
                message="matrix-rank result exceeds the canonical output bound",
            )
    execution_checkpoint("after result admission")
    # Compute the exact deterministic pivots using the maintained backend.
    data = _compute_matrix_rank_kernel(
        matrix,
        execution_checkpoint=execution_checkpoint,
    )
    result = MatrixRankResult(
        matrix=matrix,
        rank=data.rank,
        pivot_rows=data.pivot_rows,
        pivot_columns=data.pivot_columns,
    )
    execution_checkpoint("after result construction")
    return result


def compute_rank(request: MatrixRankRequest) -> MatrixRankResult:
    """Compute rank through the typed wire request adapter."""

    return compute_matrix_rank(request.matrix)


def _run_matrix_rank(request: MatrixRankRequest) -> MatrixRankResult:
    """Run matrix rank through the canonical delivery boundary."""

    return compute_matrix_rank(request.matrix, enforce_transport_limit=True)


MATRIX_RANK_OPERATION = MathTool(
    operation_id="finite_field.matrix.rank.compute",
    title="Compute exact rank of a labelled matrix over its presented field",
    description=(
        "Given one AxisBoundMatrix bound to a FiniteFieldPresentation, return its "
        "exact rank over that field with deterministic row and column pivot labels. "
        "Supports both prime and extension fields."
    ),
    request_type=MatrixRankRequest,
    result_type=MatrixRankResult,
    run=_run_matrix_rank,
    tags=("finite-field", "matrix", "rank", "exact"),
    examples=(
        example(
            "rank_one_over_f2",
            "Rank [[1,1],[1,1]] over F_2 is 1; the matrix must use one consistent field presentation.",
            {"matrix": _MATRIX},
        ),
    ),
)


__all__ = ["MATRIX_RANK_OPERATION", "compute_matrix_rank", "compute_rank"]
