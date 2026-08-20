"""Polynomial interpolation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomial_interpolation_ops._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonFormRequest,
    NewtonFormResult,
)
from jacobian.math.polynomial_interpolation_ops._operations import (
    compute_divided_differences,
    compute_newton_evaluate,
    compute_newton_form,
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
        "polynomial.interpolation.divided_differences.compute",
        "Compute Newton divided differences",
        "Compute the divided differences table from sample points using "
        "exact rational arithmetic via SymPy.",
        DividedDifferencesRequest,
        DividedDifferencesResult,
        compute_divided_differences,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "three_points",
                "Divided differences for points (0,1), (1,2), (2,5).",
                {
                    "nodes": ["0", "1", "2"],
                    "values": ["1", "2", "5"],
                },
            ),
        ),
    ),
    _op(
        "polynomial.interpolation.newton_form.compute",
        "Compute Newton form of the interpolating polynomial",
        "Compute the Newton form coefficients from sample points using "
        "exact rational arithmetic.",
        NewtonFormRequest,
        NewtonFormResult,
        compute_newton_form,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "three_points",
                "Newton form for points (0,1), (1,2), (2,5).",
                {
                    "nodes": ["0", "1", "2"],
                    "values": ["1", "2", "5"],
                },
            ),
        ),
    ),
    _op(
        "polynomial.interpolation.newton_evaluate.compute",
        "Evaluate a polynomial in Newton form at a point",
        "Evaluate the interpolating polynomial in Newton form at a given "
        "point using nested multiplication.",
        NewtonEvaluateRequest,
        NewtonEvaluateResult,
        compute_newton_evaluate,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "evaluate_at_3",
                "Evaluate the interpolant of (0,1), (1,2), (2,5) at x=3.",
                {
                    "nodes": ["0", "1", "2"],
                    "values": ["1", "2", "5"],
                    "evaluation_point": "3",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
