"""Polynomial interpolation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.polynomial_interpolation.operations import (
    compute_multipoint_evaluate,
    compute_newton_interpolation,
)
from jacobian.math.polynomial_interpolation import (
    MultipointEvaluationRequest,
    MultipointEvaluationResult,
    NewtonInterpolationRequest,
    NewtonInterpolationResult,
)
from jacobian.math_tools import MathTool


def pi_operation[RequestT: ContractModel, ResultT: ContractModel](
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


def _run_newton(request: NewtonInterpolationRequest) -> NewtonInterpolationResult:
    return compute_newton_interpolation(request)


def _run_multipoint(request: MultipointEvaluationRequest) -> MultipointEvaluationResult:
    return compute_multipoint_evaluate(request)


POLYNOMIAL_INTERPOLATION_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    pi_operation(
        "polynomial.newton_interpolation.compute",
        "Newton-form interpolation via divided differences",
        "Compute the exact interpolating polynomial through given rational "
        "data points using Newton's divided difference method. Returns both "
        "standard coefficients and the divided differences.",
        NewtonInterpolationRequest,
        NewtonInterpolationResult,
        _run_newton,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "two_points",
                "Interpolate through (0,1) and (1,2): polynomial x + 1.",
                {
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "2", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    pi_operation(
        "polynomial.multipoint_evaluate.compute",
        "Evaluate a polynomial at multiple points",
        "Evaluate a univariate polynomial at multiple rational points "
        "simultaneously using Horner's method for each point.",
        MultipointEvaluationRequest,
        MultipointEvaluationResult,
        _run_multipoint,
        "polynomial",
        "evaluation",
        "exact",
        examples=(
            example(
                "quadratic",
                "Evaluate x^2 + 1 at x=0, x=1, x=2: [1, 2, 5].",
                {
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "evaluation_points": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)
