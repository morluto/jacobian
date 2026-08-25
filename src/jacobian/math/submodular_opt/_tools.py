"""Submodular optimization operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.submodular_opt._models import (
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)
from jacobian.math.submodular_opt._operations import (
    check_monotonicity,
    check_submodularity,
    evaluate_set_function,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


SUBMODULAR_OPT_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "combinatorics.set_function.evaluate",
        "Evaluate a set function",
        "Evaluate f(S) by table lookup.",
        SetFunctionEvalRequest,
        SetFunctionEvalResult,
        evaluate_set_function,
        "combinatorics",
        "set-function",
        "exact",
        examples=(
            example(
                "simple_eval",
                "Evaluate a simple set function.",
                {
                    "function": {
                        "ground_set_size": 2,
                        "entries": [
                            {"subset": [], "value": {"num": "0", "den": "1"}},
                            {"subset": [0], "value": {"num": "1", "den": "1"}},
                            {"subset": [1], "value": {"num": "1", "den": "1"}},
                            {"subset": [0, 1], "value": {"num": "2", "den": "1"}},
                        ],
                    },
                    "subset": [0, 1],
                },
            ),
        ),
    ),
    _op(
        "combinatorics.set_function.monotonicity",
        "Check monotonicity of a set function",
        "Check if a set function is monotone non-decreasing by scanning every "
        "covering relation. Scan admission bounds each table value to 128 "
        "numerator/denominator digits and the complete 2^n table to the 9 MiB "
        "transport envelope; ground sets up to 16 elements.",
        MonotonicityCheckRequest,
        MonotonicityCheckResult,
        check_monotonicity,
        "combinatorics",
        "set-function",
        "monotonicity",
        "exact",
        examples=(
            example(
                "monotone_check",
                "Check if a set function is monotone.",
                {
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
    _op(
        "combinatorics.set_function.submodularity",
        "Check submodularity of a set function",
        "Check if a set function is submodular via the exact local "
        "characterization f(S+i)+f(S+j) >= f(S)+f(S+{i,j}) over all pairs. Scan "
        "admission bounds each table value to 128 numerator/denominator digits "
        "and the complete 2^n table to the 9 MiB transport envelope; ground "
        "sets up to 16 elements.",
        SubmodularityCheckRequest,
        SubmodularityCheckResult,
        check_submodularity,
        "combinatorics",
        "set-function",
        "submodularity",
        "exact",
        examples=(
            example(
                "submodular_check",
                "Check if a set function is submodular.",
                {
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


TOOLS = SUBMODULAR_OPT_OPERATIONS

__all__ = ["TOOLS"]
