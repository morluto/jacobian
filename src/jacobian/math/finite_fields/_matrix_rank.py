"""Finite-field matrix rank operation declaration."""

from __future__ import annotations

from functools import partial
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
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


def _run_matrix_rank(request: MatrixRankRequest) -> MatrixRankResult:
    """Run matrix rank through the canonical delivery boundary."""

    return compute_matrix_rank(request.matrix)


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
        OperationExample(
            name="rank_one_over_f2",
            description="Rank [[1,1],[1,1]] over F_2 is 1; the matrix must use one consistent field presentation.",
            input={"matrix": _MATRIX},
        ),
    ),
)


__all__ = ["MATRIX_RANK_OPERATION", "compute_matrix_rank", "compute_rank"]
