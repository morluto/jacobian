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
    GenericDegreeRequest,
    GenericDegreeResult,
    JacobianRequest,
    JacobianResult,
)
from jacobian.math.polynomials.maps._operations import (
    compose_polynomials,
    compute_generic_degree,
    compute_jacobian,
    evaluate_polynomial,
)


def _polynomial(
    variable: str,
    *terms: tuple[int, int],
) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
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
        "polynomial_schema_version": "1",
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
    version: str = "2",
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
        "polynomial.map.generic_degree.compute",
        "Compute the exact generic degree of a polynomial map",
        "Classify the generic scheme-theoretic fiber of a bounded polynomial "
        "map over QQ and, when it is finite, return its exact quotient dimension "
        "with source-bound Groebner evidence. This computes over the generic "
        "target function field and never infers degree from a sampled fiber.",
        GenericDegreeRequest,
        GenericDegreeResult,
        compute_generic_degree,
        "polynomial",
        "algebraic-geometry",
        "generic-fiber",
        "exact",
        examples=(
            example(
                "quadratic_generic_degree",
                "Compute generic degree 2 for (x, y) -> (x^2, y); every map "
                "component must use the complete ordered source axis.",
                {
                    "polynomial_map": {
                        "polynomial_map_schema_version": "1",
                        "input_variables": ["x", "y"],
                        "output_polynomials": [
                            _bivariate_polynomial((1, (2, 0))),
                            _bivariate_polynomial((1, (0, 1))),
                        ],
                    }
                },
            ),
        ),
        version="1",
    ),
    _op(
        "polynomial.map.evaluate",
        "Evaluate a polynomial at a rational point",
        "Evaluate a canonical rational polynomial at a complete ordered rational point.",
        EvalRequest,
        EvalResult,
        evaluate_polynomial,
        "polynomial",
        "evaluation",
        "exact",
        examples=(
            example(
                "simple_eval",
                "Evaluate x^2 + 2y at x=3, y=1; the point must use the "
                "polynomial's complete ordered axis.",
                {
                    "polynomial": _bivariate_polynomial(
                        (1, (2, 0)),
                        (2, (0, 1)),
                    ),
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
        "Compute the row-major Jacobian matrix of a canonical polynomial map.",
        JacobianRequest,
        JacobianResult,
        compute_jacobian,
        "polynomial",
        "jacobian",
        "exact",
        examples=(
            example(
                "simple_jacobian",
                "Compute the Jacobian of [x^2, y^2] with respect to (x, y); "
                "every output must use that complete ordered axis.",
                {
                    "input_variables": ["x", "y"],
                    "output_polynomials": [
                        _bivariate_polynomial((1, (2, 0))),
                        _bivariate_polynomial((1, (0, 2))),
                    ],
                },
            ),
        ),
    ),
    _op(
        "polynomial.map.compose",
        "Compose two polynomials",
        "Compose two bounded univariate canonical rational polynomials.",
        CompositionRequest,
        CompositionResult,
        compose_polynomials,
        "polynomial",
        "composition",
        "exact",
        examples=(
            example(
                "simple_compose",
                "Compose x^2 with x+1; each polynomial must use exactly its "
                "declared substitution variable.",
                {
                    "outer": _polynomial("x", (1, 2)),
                    "inner": _polynomial("x", (1, 1), (1, 0)),
                    "inner_variable": "x",
                    "outer_variable": "x",
                },
            ),
        ),
    ),
)


TOOLS = POLYNOMIAL_MAP_OPERATIONS

__all__ = ["TOOLS"]
