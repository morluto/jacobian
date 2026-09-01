"""Polynomial map operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    GenericDegreeRequest,
    GenericDegreeResult,
    JacobianResult,
)
from jacobian.math.polynomials.maps.operations import (
    compose_polynomials,
    evaluate_polynomial,
    generic_degree,
    jacobian_matrix,
)
from jacobian.math.polynomials.maps.values import RationalPolynomialMap


def _generic_degree(request: GenericDegreeRequest) -> GenericDegreeResult:
    return generic_degree(request.polynomial_map, request.resource_budget)


def _evaluate(request: EvalRequest) -> EvalResult:
    return evaluate_polynomial(request.polynomial, request.point)


def _compose(request: CompositionRequest) -> CompositionResult:
    return compose_polynomials(
        request.outer,
        request.inner,
        outer_variable=request.outer_variable,
        inner_variable=request.inner_variable,
    )


def _polynomial(
    variable: str,
    *terms: tuple[int, int],
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": [variable],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": [exponent],
                }
                for coefficient, exponent in terms
            ]
        },
    }


def _bivariate_polynomial(*terms: tuple[int, tuple[int, int]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": list(exponents),
                }
                for coefficient, exponents in terms
            ]
        },
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.map.generic_degree.compute",
        title="Compute the exact generic degree of a polynomial map",
        description="Classify the generic scheme-theoretic fiber of a bounded polynomial "
        "map over QQ and, when it is finite, return its exact quotient dimension "
        "with source-bound Groebner evidence. This computes over the generic "
        "target function field and never infers degree from a sampled fiber.",
        request_type=GenericDegreeRequest,
        result_type=GenericDegreeResult,
        run=_generic_degree,
        tags=("polynomial", "algebraic-geometry", "generic-fiber", "exact"),
        examples=(
            OperationExample(
                name="quadratic_generic_degree",
                description="Compute generic degree 2 for (x, y) -> (x^2, y); every map "
                "component must use the complete ordered source axis.",
                input={
                    "polynomial_map": {
                        "input_variables": ["x", "y"],
                        "output_polynomials": [
                            _bivariate_polynomial((1, (2, 0))),
                            _bivariate_polynomial((1, (0, 1))),
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.map.evaluate",
        title="Evaluate a polynomial at a rational point",
        description="Evaluate a canonical rational polynomial at a complete ordered "
        "rational point.",
        request_type=EvalRequest,
        result_type=EvalResult,
        run=_evaluate,
        tags=("polynomial", "evaluation", "exact"),
        examples=(
            OperationExample(
                name="simple_eval",
                description="Evaluate x^2 + 2y at x=3, y=1.",
                input={
                    "polynomial": _bivariate_polynomial(
                        (1, (2, 0)),
                        (2, (0, 1)),
                    ),
                    "point": {
                        "variables": ["x", "y"],
                        "values": [
                            {"num": "3", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.map.jacobian",
        title="Compute the Jacobian matrix of a polynomial map",
        description="Compute the row-major Jacobian matrix of a canonical polynomial map.",
        request_type=RationalPolynomialMap,
        result_type=JacobianResult,
        run=jacobian_matrix,
        tags=("polynomial", "jacobian", "exact"),
        examples=(
            OperationExample(
                name="simple_jacobian",
                description="Compute the Jacobian of [x^2, y^2] with respect to (x, y); "
                "every output must use that complete ordered axis.",
                input={
                    "input_variables": ["x", "y"],
                    "output_polynomials": [
                        _bivariate_polynomial((1, (2, 0))),
                        _bivariate_polynomial((1, (0, 2))),
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.map.compose",
        title="Compose two univariate polynomials",
        description="Compute the exact composition of two bounded univariate canonical "
        "rational polynomials.",
        request_type=CompositionRequest,
        result_type=CompositionResult,
        run=_compose,
        tags=("polynomial", "composition", "exact"),
        examples=(
            OperationExample(
                name="simple_compose",
                description="Compose x^2 with x+1.",
                input={
                    "outer": _polynomial("x", (1, 2)),
                    "inner": _polynomial("x", (1, 1), (1, 0)),
                    "inner_variable": "x",
                    "outer_variable": "x",
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
