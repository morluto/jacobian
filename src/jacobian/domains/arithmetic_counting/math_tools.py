"""Arithmetic counting operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.arithmetic_counting import (
    CongruenceBoxCountRequest,
    CongruenceBoxCountResult,
    FloorSumRequest,
    FloorSumResult,
)
from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.arithmetic_counting.operations import (
    compute_congruence_box_count,
    compute_floor_sum,
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


ARITHMETIC_COUNTING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "integer.counting.floor_sum.compute",
        "Compute floor sum",
        "Compute sum_{i=0}^{n-1} floor((a*i + b) / m) for bounded "
        "non-negative integers n, m, a, b.",
        FloorSumRequest,
        FloorSumResult,
        compute_floor_sum,
        "integer",
        "floor-sum",
        "exact",
        examples=(
            example(
                "simple_floor_sum",
                "floor_sum(5, 3, 2, 1).",
                {"n": 5, "m": 3, "a": 2, "b": 1},
            ),
        ),
    ),
    _op(
        "integer.counting.congruence_box.compute",
        "Count congruence-constrained lattice points",
        "Count lattice points in a bounded box satisfying u*x + v*y = c (mod modulus).",
        CongruenceBoxCountRequest,
        CongruenceBoxCountResult,
        compute_congruence_box_count,
        "integer",
        "congruence",
        "lattice-point",
        "exact",
        examples=(
            example(
                "simple_congruence",
                "Count (x+y)=0 mod 3 in [0,5]^2.",
                {
                    "x_lo": 0,
                    "x_hi": 5,
                    "y_lo": 0,
                    "y_hi": 5,
                    "u": 1,
                    "v": 1,
                    "c": 0,
                    "modulus": 3,
                },
            ),
        ),
    ),
)


__all__ = ["ARITHMETIC_COUNTING_OPERATIONS"]
