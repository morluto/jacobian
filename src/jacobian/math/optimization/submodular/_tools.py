"""Submodular optimization operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.optimization.submodular._models import (
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)
from jacobian.math.optimization.submodular.operations import (
    check_monotonicity,
    check_submodularity,
    evaluate_set_function,
)


def _evaluate(request: SetFunctionEvalRequest) -> SetFunctionEvalResult:
    return evaluate_set_function(request.function, request.subset)


def _monotonicity(request: MonotonicityCheckRequest) -> MonotonicityCheckResult:
    return check_monotonicity(request.function)


def _submodularity(request: SubmodularityCheckRequest) -> SubmodularityCheckResult:
    return check_submodularity(request.function)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorics.set_function.evaluate",
        title="Evaluate a finite set function",
        description="Look up the exact canonical rational value of a finite set function "
        "at a specified subset of its complete table.",
        request_type=SetFunctionEvalRequest,
        result_type=SetFunctionEvalResult,
        run=_evaluate,
        tags=("combinatorics", "set-function", "evaluation", "exact"),
        examples=(
            OperationExample(
                name="evaluate_singleton",
                description="Evaluate a two-element set function at the singleton {0}.",
                input={
                    "function": {
                        "ground_set_size": 1,
                        "entries": [
                            {"subset": [], "value": {"num": "0", "den": "1"}},
                            {"subset": [0], "value": {"num": "1", "den": "1"}},
                        ],
                    },
                    "subset": [0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.set_function.monotonicity",
        title="Check monotonicity of a set function",
        description="Check if a set function is monotone non-decreasing by scanning every "
        "covering relation. Scan admission bounds each table value to 128 "
        "numerator/denominator digits and the complete table to 2^n entries; "
        "ground sets have at most 16 elements.",
        request_type=MonotonicityCheckRequest,
        result_type=MonotonicityCheckResult,
        run=_monotonicity,
        tags=("combinatorics", "set-function", "monotonicity", "exact"),
        examples=(
            OperationExample(
                name="monotone_check",
                description="Check if a set function is monotone.",
                input={
                    "function": {
                        "ground_set_size": 2,
                        "entries": [
                            {"subset": [], "value": {"num": "0", "den": "1"}},
                            {"subset": [0], "value": {"num": "1", "den": "1"}},
                            {"subset": [1], "value": {"num": "1", "den": "1"}},
                            {"subset": [0, 1], "value": {"num": "2", "den": "1"}},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.set_function.submodularity",
        title="Check submodularity of a set function",
        description="Check if a set function is submodular via the exact local "
        "characterization f(S+i)+f(S+j) >= f(S)+f(S+{i,j}) over all pairs. Scan "
        "admission bounds each table value to 128 numerator/denominator digits "
        "and the complete table to 2^n entries; ground sets have at most 16 "
        "elements.",
        request_type=SubmodularityCheckRequest,
        result_type=SubmodularityCheckResult,
        run=_submodularity,
        tags=("combinatorics", "set-function", "submodularity", "exact"),
        examples=(
            OperationExample(
                name="submodular_check",
                description="Check if a set function is submodular.",
                input={
                    "function": {
                        "ground_set_size": 2,
                        "entries": [
                            {"subset": [], "value": {"num": "0", "den": "1"}},
                            {"subset": [0], "value": {"num": "1", "den": "1"}},
                            {"subset": [1], "value": {"num": "1", "den": "1"}},
                            {"subset": [0, 1], "value": {"num": "2", "den": "1"}},
                        ],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
