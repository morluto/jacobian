"""Finite-field matrix rank operation declaration."""

from __future__ import annotations

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.finite_fields._matrix_rank_models import (
    MatrixRankRequest,
    MatrixRankResult,
)
from jacobian.math.finite_fields.operations import matrix_rank

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
    try:
        active_rows = tuple(
            label
            for index, label in enumerate(matrix.row_axis.labels)
            if any(not element.is_zero for element in matrix.entries[index])
        )
        active_columns = tuple(
            label
            for index, label in enumerate(matrix.column_axis.labels)
            if any(not row[index].is_zero for row in matrix.entries)
        )
        rank_bound = min(len(active_rows), len(active_columns))
        result_probe = encode_strict_json(
            {
                "matrix": matrix.model_dump(mode="json"),
                "rank": rank_bound,
                "pivot_rows": sorted(
                    active_rows,
                    key=lambda label: len(encode_strict_json(label)),
                    reverse=True,
                )[:rank_bound],
                "pivot_columns": sorted(
                    active_columns,
                    key=lambda label: len(encode_strict_json(label)),
                    reverse=True,
                )[:rank_bound],
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
    return matrix_rank(matrix)


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
