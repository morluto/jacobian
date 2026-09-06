"""Exact rank of a labelled matrix over its presented finite field."""

from __future__ import annotations

from functools import partial
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.math.finite_fields._matrix_rank_models import (
    MatrixRankRequest,
    MatrixRankResult,
)
from jacobian.math.finite_fields.values import Axis, AxisBoundMatrix

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
    request_checkpoint(stage)
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"finite-field rank deadline expired {stage}"
        )


def compute_matrix_rank(
    matrix: AxisBoundMatrix,
) -> MatrixRankResult:
    """Return the exact rank of a labelled matrix over its presented finite field."""
    deadline = _execution_deadline()
    execution_checkpoint = partial(_require_deadline, deadline)
    execution_checkpoint("after admission")
    # Compute the exact deterministic pivots using the maintained backend.
    from jacobian.math.finite_fields._matrix_rank_kernels import (
        compute_matrix_rank as _compute_matrix_rank_kernel,
    )

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


def verify_matrix_rank(claim: MatrixRankResult) -> bool:
    """Check rank equality and nonsingularity of the declared pivot minor.

    The native axis carrier bounds each axis by 1024 and the field order by
    65536. Two eliminations cost at most twice 1024 cubed field operations.
    They share one field admission and one execution deadline. Any full-rank
    minor is accepted, independently of the producer's pivot selection.
    """
    from jacobian.math.finite_fields._flint import context
    from jacobian.math.finite_fields._matrix_rank_kernels import (
        compute_matrix_rank as kernel_rank,
    )

    checkpoint = partial(_require_deadline, _execution_deadline())
    checkpoint("before rank claim admission")
    matrix = claim.matrix
    active_context = context(matrix.presentation)
    actual = kernel_rank(
        matrix, execution_checkpoint=checkpoint, active_context=active_context
    )
    if actual.rank != claim.rank:
        return False
    rows = {label: i for i, label in enumerate(matrix.row_axis.labels)}
    columns = {label: i for i, label in enumerate(matrix.column_axis.labels)}
    minor = AxisBoundMatrix(
        presentation=matrix.presentation,
        row_axis=Axis(name=matrix.row_axis.name, labels=claim.pivot_rows),
        column_axis=Axis(name=matrix.column_axis.name, labels=claim.pivot_columns),
        entries=tuple(
            tuple(matrix.entries[rows[r]][columns[c]] for c in claim.pivot_columns)
            for r in claim.pivot_rows
        ),
    )
    minor_rank = kernel_rank(
        minor, execution_checkpoint=checkpoint, active_context=active_context
    )
    checkpoint("after rank claim verification")
    return minor_rank.rank == claim.rank


__all__ = ["compute_matrix_rank", "compute_rank", "verify_matrix_rank"]
