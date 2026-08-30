"""Finite-field matrix rank operation declaration."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
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
    data = compute_matrix_rank(request.matrix)
    return MatrixRankResult(
        matrix=request.matrix,
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


__all__ = ["compute_rank", "MATRIX_RANK_OPERATION"]
