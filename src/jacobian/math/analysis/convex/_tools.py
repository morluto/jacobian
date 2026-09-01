"""Convex analysis operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.convex._models import (
    MaxAffineEvalRequest,
    MaxAffineEvalResult,
    MaxAffineSubdifferentialRequest,
    MaxAffineSubdifferentialResult,
)
from jacobian.math.analysis.convex.operations import (
    max_affine_evaluation,
    max_affine_subdifferential,
)


def _run_max_affine_evaluation(request: MaxAffineEvalRequest) -> MaxAffineEvalResult:
    return max_affine_evaluation(request.function, request.point)


def _run_subdifferential(
    request: MaxAffineSubdifferentialRequest,
) -> MaxAffineSubdifferentialResult:
    return max_affine_subdifferential(request.function, request.point)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="convex.max_affine.evaluate",
        title="Evaluate a max-affine function",
        description="Evaluate f(x) = max_i { <a_i, x> + b_i } at a rational point "
        "and identify all active pieces.",
        request_type=MaxAffineEvalRequest,
        result_type=MaxAffineEvalResult,
        run=_run_max_affine_evaluation,
        tags=("convex", "max-affine", "exact"),
        examples=(
            OperationExample(
                name="simple_max",
                description="max(x, -x) at x=2.",
                input={
                    "function": {
                        "pieces": [
                            {
                                "piece_id": "p1",
                                "coefficients": [{"num": "1", "den": "1"}],
                                "intercept": {"num": "0", "den": "1"},
                            },
                            {
                                "piece_id": "p2",
                                "coefficients": [{"num": "-1", "den": "1"}],
                                "intercept": {"num": "0", "den": "1"},
                            },
                        ],
                    },
                    "point": {"coordinates": [{"num": "2", "den": "1"}]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="convex.max_affine.subdifferential",
        title="Compute subdifferential of a max-affine function",
        description="Compute the subdifferential (set of active gradients) of a "
        "max-affine function at a rational point.",
        request_type=MaxAffineSubdifferentialRequest,
        result_type=MaxAffineSubdifferentialResult,
        run=_run_subdifferential,
        tags=("convex", "subdifferential", "exact"),
        examples=(
            OperationExample(
                name="simple_subdiff",
                description="Subdifferential of max(x, -x) at x=2.",
                input={
                    "function": {
                        "pieces": [
                            {
                                "piece_id": "p1",
                                "coefficients": [{"num": "1", "den": "1"}],
                                "intercept": {"num": "0", "den": "1"},
                            },
                            {
                                "piece_id": "p2",
                                "coefficients": [{"num": "-1", "den": "1"}],
                                "intercept": {"num": "0", "den": "1"},
                            },
                        ],
                    },
                    "point": {"coordinates": [{"num": "2", "den": "1"}]},
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
