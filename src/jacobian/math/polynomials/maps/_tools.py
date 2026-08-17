"""Polynomial map operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    JacobianRequest,
    JacobianResult,
)
from jacobian.math.polynomials.maps._operations import (
    compose_polynomials,
    compute_jacobian,
    evaluate_polynomial,
)


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


POLYNOMIAL_MAP_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polynomial.map.evaluate",
        "Evaluate a polynomial at a rational point",
        "Evaluate a polynomial expression at a rational point using SymPy.",
        EvalRequest,
        EvalResult,
        evaluate_polynomial,
        "polynomial",
        "evaluation",
        "exact",
        examples=(
            example(
                "simple_eval",
                "Evaluate x**2 + 2*y at x=3, y=1.",
                {
                    "polynomial": {"expression": "x**2 + 2*y"},
                    "point": {
                        "variables": ["x", "y"],
                        "values": [{"num": "3", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                },
            ),
        ),
    ),
    _op(
        "polynomial.map.jacobian",
        "Compute the Jacobian matrix of a polynomial map",
        "Compute the Jacobian matrix dF/dx of a polynomial map using SymPy.",
        JacobianRequest,
        JacobianResult,
        compute_jacobian,
        "polynomial",
        "jacobian",
        "exact",
        examples=(
            example(
                "simple_jacobian",
                "Jacobian of [x**2, y**2] w.r.t. (x, y).",
                {
                    "input_variables": ["x", "y"],
                    "output_polynomials": [
                        {"expression": "x**2"},
                        {"expression": "y**2"},
                    ],
                },
            ),
        ),
    ),
    _op(
        "polynomial.map.compose",
        "Compose two polynomials",
        "Compose outer(inner(x)) using SymPy substitution.",
        CompositionRequest,
        CompositionResult,
        compose_polynomials,
        "polynomial",
        "composition",
        "exact",
        examples=(
            example(
                "simple_compose",
                "Compose x**2 with x+1.",
                {
                    "outer": {"expression": "x**2"},
                    "inner": {"expression": "x + 1"},
                    "inner_variable": "x",
                    "outer_variable": "x",
                },
            ),
        ),
    ),
)


TOOLS = POLYNOMIAL_MAP_OPERATIONS

__all__ = ["TOOLS"]
