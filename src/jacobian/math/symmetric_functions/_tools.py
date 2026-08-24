"""Symmetric function operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.symmetric_functions._models import (
    SchurExpansionRequest,
    SchurExpansionResult,
)
from jacobian.math.symmetric_functions._operations import compute_schur_evaluation


def sf_op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    sf_op(
        "symmetric_function.schur.evaluate.compute",
        "Evaluate a Schur function at a point",
        "Evaluate the Schur function s_lambda(x_1,...,x_n) at a bounded "
        "integer point using the Jacobi-Trudi determinant formula.",
        SchurExpansionRequest,
        SchurExpansionResult,
        compute_schur_evaluation,
        "symmetric-function",
        "schur",
        "exact",
        examples=(
            example(
                "schur_1_at_1_1",
                "Evaluate s_(1)(1,1) = 2. Needs: decreasing positive parts "
                "with total size <=500 and at most 50 parts; distinct "
                "variables whose count equals the point length (both "
                "1..20); |coordinate| <=999999.",
                {
                    "partition": {"parts": [1]},
                    "variables": ["x1", "x2"],
                    "point": [1, 1],
                },
            ),
        ),
        version="2",
    ),
)

__all__ = ["TOOLS"]
