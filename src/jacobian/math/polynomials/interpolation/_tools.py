"""Polynomial interpolation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.interpolation._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    HermiteInterpolationRequest,
    HermiteInterpolationResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonForm,
    NewtonFormRequest,
)
from jacobian.math.polynomials.interpolation.operations import (
    divided_differences,
    evaluate_newton,
    hermite_interpolation,
    newton_form,
)


def _hermite(request: HermiteInterpolationRequest) -> HermiteInterpolationResult:
    return hermite_interpolation(request.table)


def _divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    return divided_differences(request.samples)


def _newton_form(request: NewtonFormRequest) -> NewtonForm:
    return newton_form(request.samples)


def _evaluate_newton(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    return evaluate_newton(request.newton_form, request.evaluation_point)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


def _rational(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _samples() -> dict[str, list[dict[str, str]]]:
    return {
        "nodes": [_rational(value) for value in (0, 1, 2)],
        "values": [_rational(value) for value in (1, 2, 5)],
    }


def _hermite_table() -> dict[str, object]:
    return {
        "variable": "x",
        "jets": [
            {
                "node": _rational(0),
                "derivatives": [
                    {"derivative_order": 0, "value": _rational(0)},
                    {"derivative_order": 1, "value": _rational(0)},
                ],
            },
            {
                "node": _rational(1),
                "derivatives": [
                    {"derivative_order": 0, "value": _rational(1)},
                    {"derivative_order": 1, "value": _rational(2)},
                ],
            },
        ],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polynomial.interpolation.hermite.compute",
        "Compute an exact rational Hermite interpolant",
        "Return the unique degree-<M polynomial in QQ[x] whose ordinary "
        "derivatives match a complete table of rational derivative jets, with "
        "a complete exact source-bound replay ledger. Uses a preflighted exact "
        "Hermite-Vandermonde solve; rows require distinct nodes and derivative "
        "orders 0 through m-1.",
        HermiteInterpolationRequest,
        HermiteInterpolationResult,
        _hermite,
        "polynomial",
        "interpolation",
        "hermite",
        "derivative-jet",
        "confluent-interpolation",
        "exact",
        examples=(
            example(
                "quadratic_from_two_first_order_jets",
                "Compute the exact polynomial matching value and first derivative "
                "jets of x^2 at 0 and 1; nodes must be distinct and each jet must "
                "list the complete derivative-order prefix starting at zero.",
                {"table": _hermite_table()},
            ),
        ),
    ),
    _op(
        "polynomial.interpolation.divided_differences.compute",
        "Compute Newton divided differences",
        "Compute the divided differences table from sample points using "
        "bounded exact rational arithmetic.",
        DividedDifferencesRequest,
        DividedDifferencesResult,
        _divided_differences,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "three_points",
                "Divided differences for points (0,1), (1,2), (2,5).",
                {"samples": _samples()},
            ),
        ),
    ),
    _op(
        "polynomial.interpolation.newton_form.compute",
        "Compute Newton form of the interpolating polynomial",
        "Compute the Newton form coefficients from sample points using "
        "exact rational arithmetic.",
        NewtonFormRequest,
        NewtonForm,
        _newton_form,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "three_points",
                "Newton form for points (0,1), (1,2), (2,5).",
                {"samples": _samples()},
            ),
        ),
    ),
    _op(
        "polynomial.interpolation.newton_evaluate.compute",
        "Evaluate a polynomial in Newton form at a point",
        "Evaluate a canonical NewtonForm directly at a rational point using "
        "nested multiplication.",
        NewtonEvaluateRequest,
        NewtonEvaluateResult,
        _evaluate_newton,
        "polynomial",
        "interpolation",
        "exact",
        examples=(
            example(
                "evaluate_at_3",
                "Evaluate the interpolant of (0,1), (1,2), (2,5) at x=3.",
                {
                    "newton_form": {
                        "nodes": [_rational(value) for value in (0, 1, 2)],
                        "coefficients": [_rational(1), _rational(1), _rational(1)],
                    },
                    "evaluation_point": _rational(3),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
