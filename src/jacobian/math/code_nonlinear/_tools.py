"""Nonlinear binary code operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_nonlinear._models import (
    BinaryCodeRequest,
    ConstantWeightRequest,
    ConstantWeightResult,
    DistanceProfileResult,
)
from jacobian.math.code_nonlinear._operations import (
    compute_constant_weight,
    compute_distance_profile,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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
    _op(
        "code.nonlinear.distance_profile.compute",
        "Compute the distance profile of a binary code",
        "Compute the minimum Hamming distance and weight profile of a nonlinear binary code by exact enumeration.",
        BinaryCodeRequest,
        DistanceProfileResult,
        compute_distance_profile,
        "code",
        "distance",
        "exact",
        examples=(
            example(
                "binary_code",
                "Distance profile of a simple binary code.",
                {"codewords": [[0, 0, 0], [1, 1, 0], [0, 1, 1]]},
            ),
        ),
    ),
    _op(
        "code.nonlinear.constant_weight.compute",
        "Generate all constant-weight binary words",
        "Generate all binary words of given length and Hamming weight.",
        ConstantWeightRequest,
        ConstantWeightResult,
        compute_constant_weight,
        "code",
        "constant-weight",
        "exact",
        examples=(
            example(
                "weight_two_length_four",
                "All weight-2 binary words of length 4.",
                {"length": 4, "weight": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
