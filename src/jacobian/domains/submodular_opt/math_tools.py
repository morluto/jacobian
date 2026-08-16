"""Submodular optimization operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.contracts.submodular_opt import (
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)
from jacobian.domains._examples import example
from jacobian.domains.submodular_opt.operations import (
    check_monotonicity,
    check_submodularity,
    evaluate_set_function,
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
        "Check if a set function is monotone non-decreasing.",
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
        "Check if a set function is submodular.",
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


__all__ = ["SUBMODULAR_OPT_OPERATIONS"]
