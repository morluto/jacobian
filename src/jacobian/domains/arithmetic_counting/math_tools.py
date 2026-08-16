"""MathTool declarations for arithmetic counting operations."""

from __future__ import annotations

from jacobian.contracts.arithmetic_counting import (
    CongruenceConstrainedCountRequest,
    CongruenceConstrainedCountResult,
    FloorSumRequest,
    FloorSumResult,
)
from jacobian.domains._examples import example
from jacobian.domains.arithmetic_counting.operations import (
    compute_congruence_constrained_count,
    compute_floor_sum,
)
from jacobian.math_tools import MathTool


ARITHMETIC_COUNTING_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="arithmetic.floor_sum.compute",
        version="1",
        title="Compute sum_{i=0}^{n-1} floor((a*i+b)/m)",
        description=(
            "Compute the floor sum sum_{i=0}^{n-1} floor((a*i+b)/m) "
            "using an O(log m) recursive algorithm."
        ),
        request_type=FloorSumRequest,
        result_type=FloorSumResult,
        run=compute_floor_sum,
        tags=(
            "arithmetic",
            "floor-sum",
            "number-theory",
            "exact",
            "lattice",
        ),
        examples=(
            example(
                "simple_floor_sum",
                "sum_{i=0}^{3} floor((2*i+1)/3) = 0+1+1+2 = 4.",
                {"n": 4, "m": 3, "a": 2, "b": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic.congruence_constrained.count",
        version="1",
        title="Count congruence-constrained lattice points",
        description=(
            "Count lattice points (b1,b2) with lower<=b1<=upper, "
            "1<=b2<=m, b1+b2>=m+1, and b2≡n*b1 mod k."
        ),
        request_type=CongruenceConstrainedCountRequest,
        result_type=CongruenceConstrainedCountResult,
        run=compute_congruence_constrained_count,
        tags=(
            "arithmetic",
            "congruence",
            "lattice",
            "counting",
            "exact",
        ),
        examples=(
            example(
                "simple_count",
                "Count for k=3, m=2, n=1, lower=1, upper=2.",
                {"k": 3, "m": 2, "n": 1, "lower": 1, "upper": 2},
            ),
        ),
    ),
)

__all__ = ["ARITHMETIC_COUNTING_OPERATIONS"]
