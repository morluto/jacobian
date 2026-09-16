"""Linear matroid operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.matroids._models import (
    MatroidClosureRequest,
    MatroidClosureResult,
    MaximumWeightBasisRequest,
    MaximumWeightBasisResult,
)
from jacobian.math.combinatorics.matroids.operations import (
    closure_result,
    maximum_weight_basis_result,
)


def _run_closure(request: MatroidClosureRequest) -> MatroidClosureResult:
    return closure_result(request.matroid, request.subset)


def _run_maximum_weight_basis(
    request: MaximumWeightBasisRequest,
) -> MaximumWeightBasisResult:
    return maximum_weight_basis_result(request.matroid, request.weights)


_CLOSURE_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "subset": [0, 1],
}

_MAXIMUM_WEIGHT_BASIS_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "weights": [3, 2, 5],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matroid.closure.compute",
        title="Compute the closure of a subset in a linear matroid",
        description=(
            "Compute the closure (smallest flat) of a subset S in a matroid "
            "represented by the columns of a canonical matrix over GF(p). "
            "The closure adds all elements that lie in the span of S."
        ),
        request_type=MatroidClosureRequest,
        result_type=MatroidClosureResult,
        run=_run_closure,
        tags=("matroid", "closure", "flat", "exact"),
        examples=(
            OperationExample(
                name="closure_of_basis",
                description="Compute the closure of {0, 1} in a rank-2 matroid.",
                input=_CLOSURE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.basis.maximum_weight.compute",
        title="Compute a maximum-weight basis of a linear matroid",
        description=(
            "Run the deterministic greedy algorithm (decreasing weight, ties "
            "by increasing index) on a represented matroid with exact integer "
            "weights keyed by the ground set. Return one maximum-weight "
            "basis, its exact total, the deterministic greedy order, and one "
            "fundamental-circuit exchange row per outside element replaying "
            "that no valid single-element exchange strictly improves the total."
        ),
        request_type=MaximumWeightBasisRequest,
        result_type=MaximumWeightBasisResult,
        run=_run_maximum_weight_basis,
        tags=("matroid", "basis", "maximum-weight", "greedy", "exact"),
        discovery_terms=(
            "maximum weight basis",
            "matroid greedy algorithm",
            "fundamental circuit",
            "basis exchange",
        ),
        examples=(
            OperationExample(
                name="triangle_weights_3_2_5",
                description="Maximum-weight basis of a rank-2 matroid picks {0, 2}.",
                input=_MAXIMUM_WEIGHT_BASIS_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
