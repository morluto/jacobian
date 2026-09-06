"""Arithmetic counting operation declarations."""

from typing import Any

from jacobian.canonical import format_canonical_integer
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
        n=request.n,
        m=request.m,
        a=request.a,
        b=request.b,
        value=format_canonical_integer(
            floor_sum(request.n, request.m, request.a, request.b)
        ),
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
    return CongruenceBoxCountResult(
        x_lo=request.x_lo,
        x_hi=request.x_hi,
        y_lo=request.y_lo,
        y_hi=request.y_hi,
        u=request.u,
        v=request.v,
        c=request.c,
        count=count,
        modulus=request.modulus,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="integer.counting.floor_sum.compute",
        title="Compute floor sum",
        description="Compute sum_{i=0}^{n-1} floor((a*i + b) / m) for bounded "
        "non-negative integers n, m, a, b.",
        request_type=FloorSumRequest,
        result_type=FloorSumResult,
        run=compute_floor_sum,
        tags=("integer", "floor-sum", "exact"),
        examples=(
            OperationExample(
                name="simple_floor_sum",
                description="floor_sum(5, 3, 2, 1).",
                input={"n": 5, "m": 3, "a": 2, "b": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.counting.congruence_box.compute",
        title="Count congruence-constrained lattice points",
        description="Count lattice points in a bounded box satisfying u*x + v*y = c (mod modulus).",
        request_type=CongruenceBoxCountRequest,
        result_type=CongruenceBoxCountResult,
        run=compute_congruence_box_count,
        tags=("integer", "congruence", "lattice-point", "exact"),
        examples=(
            OperationExample(
                name="simple_congruence",
                description="Count (x+y)=0 mod 3 in [0,5]^2.",
                input={
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
