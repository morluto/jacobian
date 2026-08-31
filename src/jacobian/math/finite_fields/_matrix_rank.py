"""Finite-field matrix rank operation declaration."""

from __future__ import annotations

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.finite_fields._matrix_rank_kernels import compute_matrix_rank
from jacobian.math.finite_fields._matrix_rank_models import (
    MatrixRankRequest,
    MatrixRankResult,
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


def compute_rank(request: MatrixRankRequest) -> MatrixRankResult:
    """Return the exact rank of a labelled matrix over its presented finite field."""
    matrix = request.matrix
    # Compute the exact deterministic pivots using the maintained backend.
    data = compute_matrix_rank(matrix)
    # Validate the complete result envelope against the canonical output bound.
    try:
        result_probe = encode_strict_json(
            {
                "matrix": matrix.model_dump(mode="json"),
                "rank": data.rank,
                "pivot_rows": list(data.pivot_rows),
                "pivot_columns": list(data.pivot_columns),
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
    return MatrixRankResult(
        matrix=matrix,
        rank=data.rank,
        pivot_rows=data.pivot_rows,
        pivot_columns=data.pivot_columns,
    )


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
    run=compute_rank,
    tags=("finite-field", "matrix", "rank", "exact"),
    examples=(
        example(
            "rank_one_over_f2",
            "Rank [[1,1],[1,1]] over F_2 is 1; the matrix must use one consistent field presentation.",
            {"matrix": _MATRIX},
        ),
    ),
)


__all__ = ["MATRIX_RANK_OPERATION", "compute_rank"]
