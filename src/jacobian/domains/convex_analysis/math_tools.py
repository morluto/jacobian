"""Convex analysis operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.convex_analysis import (
    MaxAffineEvalRequest,
    MaxAffineEvalResult,
    MaxAffineSubdifferentialRequest,
    MaxAffineSubdifferentialResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.convex_analysis.operations import (
    compute_max_affine_evaluation,
    compute_subdifferential,
)
from jacobian.math_tools import MathTool


def _op[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
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


CONVEX_ANALYSIS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "convex.max_affine.evaluate",
        "Evaluate a max-affine function",
        "Evaluate f(x) = max_i { <a_i, x> + b_i } at a rational point "
        "and identify all active pieces.",
        MaxAffineEvalRequest,
        MaxAffineEvalResult,
        compute_max_affine_evaluation,
        "convex",
        "max-affine",
        "exact",
        examples=(
            example(
                "simple_max",
                "max(x, -x) at x=2.",
                {
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
    _op(
        "convex.max_affine.subdifferential",
        "Compute subdifferential of a max-affine function",
        "Compute the subdifferential (set of active gradients) of a "
        "max-affine function at a rational point.",
        MaxAffineSubdifferentialRequest,
        MaxAffineSubdifferentialResult,
        compute_subdifferential,
        "convex",
        "subdifferential",
        "exact",
        examples=(
            example(
                "simple_subdiff",
                "Subdifferential of max(x, -x) at x=2.",
                {
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


__all__ = ["CONVEX_ANALYSIS_OPERATIONS"]
