"""Arithmetic counting operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.counting._models import (
    CongruenceBoxCountRequest,
    CongruenceBoxCountResult,
    FloorSumRequest,
    FloorSumResult,
)
from jacobian.math.number_theory.counting.operations import (
    congruence_box_count,
    floor_sum,
)


def compute_floor_sum(request: FloorSumRequest) -> FloorSumResult:
    return FloorSumResult(
        value=format_canonical_integer(
            floor_sum(request.n, request.m, request.a, request.b)
        )
    )


def compute_congruence_box_count(
    request: CongruenceBoxCountRequest,
) -> CongruenceBoxCountResult:
    count = congruence_box_count(
        x_lo=request.x_lo,
        x_hi=request.x_hi,
        y_lo=request.y_lo,
        y_hi=request.y_hi,
        u=request.u,
        v=request.v,
        c=request.c,
        modulus=request.modulus,
    )
    return CongruenceBoxCountResult(count=count, modulus=request.modulus)


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
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


__all__ = ["TOOLS"]
