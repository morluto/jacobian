"""Polynomial interpolation operation declarations."""

from typing import Any

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
    MathTool(
        operation_id="polynomial.interpolation.hermite.compute",
        title="Compute an exact rational Hermite interpolant",
        description="Return the unique degree-<M polynomial in QQ[x] whose ordinary "
        "derivatives match a complete table of rational derivative jets, with "
        "a complete exact source-bound replay ledger. Uses a preflighted exact "
        "Hermite-Vandermonde solve; rows require distinct nodes and derivative "
        "orders 0 through m-1.",
        request_type=HermiteInterpolationRequest,
        result_type=HermiteInterpolationResult,
        run=_hermite,
        tags=(
            "polynomial",
            "interpolation",
            "hermite",
            "derivative-jet",
            "confluent-interpolation",
            "exact",
        ),
        examples=(
            OperationExample(
                name="quadratic_from_two_first_order_jets",
                description="Compute the exact polynomial matching value and first derivative "
                "jets of x^2 at 0 and 1; nodes must be distinct and each jet must "
                "list the complete derivative-order prefix starting at zero.",
                input={"table": _hermite_table()},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.interpolation.divided_differences.compute",
        title="Compute Newton divided differences",
        description="Compute the divided differences table from sample points using "
        "bounded exact rational arithmetic.",
        request_type=DividedDifferencesRequest,
        result_type=DividedDifferencesResult,
        run=_divided_differences,
        tags=("polynomial", "interpolation", "exact"),
        examples=(
            OperationExample(
                name="three_points",
                description="Divided differences for points (0,1), (1,2), (2,5).",
                input={"samples": _samples()},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.interpolation.newton_form.compute",
        title="Compute Newton form of the interpolating polynomial",
        description="Compute the Newton form coefficients from sample points using "
        "exact rational arithmetic.",
        request_type=NewtonFormRequest,
        result_type=NewtonForm,
        run=_newton_form,
        tags=("polynomial", "interpolation", "exact"),
        examples=(
            OperationExample(
                name="three_points",
                description="Newton form for points (0,1), (1,2), (2,5).",
                input={"samples": _samples()},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.interpolation.newton_evaluate.compute",
        title="Evaluate a polynomial in Newton form at a point",
        description="Evaluate a canonical NewtonForm directly at a rational point using "
        "nested multiplication.",
        request_type=NewtonEvaluateRequest,
        result_type=NewtonEvaluateResult,
        run=_evaluate_newton,
        tags=("polynomial", "interpolation", "exact"),
        examples=(
            OperationExample(
                name="evaluate_at_3",
                description="Evaluate the interpolant of (0,1), (1,2), (2,5) at x=3.",
                input={
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
