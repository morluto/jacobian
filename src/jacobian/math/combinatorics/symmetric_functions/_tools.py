"""Symmetric function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.symmetric_functions._models import (
    SchurExpansionRequest,
    SchurExpansionResult,
)
from jacobian.math.combinatorics.symmetric_functions.operations import schur_evaluation


def _run_schur_evaluation(request: SchurExpansionRequest) -> SchurExpansionResult:
    return schur_evaluation(request.partition, request.point)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="symmetric_function.schur.evaluate.compute",
        title="Evaluate a Schur function at a point",
        description="Evaluate the Schur function s_lambda(x_1,...,x_n) at a bounded "
        "integer point using the Jacobi-Trudi determinant formula.",
        request_type=SchurExpansionRequest,
        result_type=SchurExpansionResult,
        run=_run_schur_evaluation,
        tags=("symmetric-function", "schur", "exact"),
        examples=(
            OperationExample(
                name="schur_1_at_1_1",
                description="Evaluate s_(1)(1,1) = 2. Needs: decreasing positive parts "
                "with total size <=500 and at most 50 parts; distinct "
                "variables whose count equals the point length (both "
                "1..20); |coordinate| <=999999.",
                input={
                    "partition": {"parts": [1]},
                    "variables": ["x1", "x2"],
                    "point": [1, 1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
