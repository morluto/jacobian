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


__all__ = ["compute_matrix_rank", "compute_rank"]
