"""Typed declarations for approximation theory operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisRequest,
    LagrangeBasisResult,
    LagrangeInterpolationRequest,
    LagrangeInterpolationResult,
)
from jacobian.math.analysis.approximation.operations import (
    lagrange_basis,
    lagrange_interpolate,
)


def _run_lagrange_basis(request: LagrangeBasisRequest) -> LagrangeBasisResult:
    return lagrange_basis(request.nodes)


def _run_lagrange_interpolation(
    request: LagrangeInterpolationRequest,
) -> LagrangeInterpolationResult:
    return LagrangeInterpolationResult(
        polynomial=lagrange_interpolate(request.nodes.nodes, request.values)
    )


def approximation_operation[
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
    approximation_operation(
        "approximation.lagrange.basis.compute",
        "Compute Lagrange basis polynomials for rational nodes",
        "Given a finite set of distinct rational nodes x_0 < ... < x_{n-1}, "
        "compute the exact Lagrange basis polynomials l_k(x) and barycentric "
        "weights w_k = 1/prod_{i!=k}(x_k - x_i) over QQ.",
        LagrangeBasisRequest,
        LagrangeBasisResult,
        _run_lagrange_basis,
        "approximation",
        "lagrange",
        "interpolation",
        "exact",
        examples=(
            example(
                "three_nodes",
                "Compute the Lagrange basis for nodes 0, 1/2, 1.",
                _BASIS_EXAMPLE,
            ),
        ),
    ),
    approximation_operation(
        "approximation.lagrange.interpolate.compute",
        "Interpolate a polynomial through rational nodes and values",
        "Given distinct rational nodes x_0, ..., x_{n-1} and values y_0, ..., "
        "y_{n-1}, compute the exact interpolation polynomial p(x) of degree at "
        "most n-1 such that p(x_k) = y_k, using the Lagrange formula over QQ.",
        LagrangeInterpolationRequest,
        LagrangeInterpolationResult,
        _run_lagrange_interpolation,
        "approximation",
        "lagrange",
        "interpolation",
        "exact",
        examples=(
            example(
                "quadratic_through_three_points",
                "Interpolate through (0,1), (1,3), (2,9).",
                _INTERP_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
