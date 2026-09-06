"""Typed declarations for approximation theory operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisRequest,
    LagrangeBasisResult,
    LagrangeInterpolationData,
    LagrangeInterpolationRequest,
    LagrangeInterpolationResult,
)
from jacobian.math.analysis.approximation.operations import (
    lagrange_basis,
    lagrange_interpolation,
)


def _run_lagrange_basis(request: LagrangeBasisRequest) -> LagrangeBasisResult:
    return lagrange_basis(request.nodes)


def _run_lagrange_interpolation(
    request: LagrangeInterpolationRequest,
) -> LagrangeInterpolationResult:
    return lagrange_interpolation(
        LagrangeInterpolationData(nodes=request.nodes, values=request.values)
    )


_BASIS_EXAMPLE: dict[str, Any] = {
    "nodes": {
        "nodes": [
            {"num": "0", "den": "1"},
            {"num": "1", "den": "2"},
            {"num": "1", "den": "1"},
        ]
    }
}

_INTERP_EXAMPLE: dict[str, Any] = {
    "nodes": {
        "nodes": [
            {"num": "0", "den": "1"},
            {"num": "1", "den": "1"},
            {"num": "2", "den": "1"},
        ]
    },
    "values": [
        {"num": "1", "den": "1"},
        {"num": "3", "den": "1"},
        {"num": "9", "den": "1"},
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="approximation.lagrange.basis.compute",
        title="Compute Lagrange basis polynomials for rational nodes",
        description="Given a finite set of distinct rational nodes x_0 < ... < x_{n-1}, "
        "compute the exact Lagrange basis polynomials l_k(x) and barycentric "
        "weights w_k = 1/prod_{i!=k}(x_k - x_i) over QQ.",
        request_type=LagrangeBasisRequest,
        result_type=LagrangeBasisResult,
        run=_run_lagrange_basis,
        tags=("approximation", "lagrange", "interpolation", "exact"),
        examples=(
            OperationExample(
                name="three_nodes",
                description="Compute the Lagrange basis for nodes 0, 1/2, 1.",
                input=_BASIS_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="approximation.lagrange.interpolate.compute",
        title="Interpolate a polynomial through rational nodes and values",
        description="Given distinct rational nodes x_0, ..., x_{n-1} and values y_0, ..., "
        "y_{n-1}, compute the exact interpolation polynomial p(x) of degree at "
        "most n-1 such that p(x_k) = y_k, using the Lagrange formula over QQ.",
        request_type=LagrangeInterpolationRequest,
        result_type=LagrangeInterpolationResult,
        run=_run_lagrange_interpolation,
        tags=("approximation", "lagrange", "interpolation", "exact"),
        examples=(
            OperationExample(
                name="quadratic_through_three_points",
                description="Interpolate through (0,1), (1,3), (2,9).",
                input=_INTERP_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
