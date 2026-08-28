"""Polynomial vector calculus operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.vector_calculus._models import (
    CurlRequest,
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
)
from jacobian.math.polynomials.vector_calculus.operations import (
    curl,
    directional_derivative,
    divergence,
    gradient,
    laplacian,
)


def _gradient(request: ScalarFieldRequest) -> VectorResult:
    return gradient(request.polynomial)


def _laplacian(request: ScalarFieldRequest) -> ScalarResult:
    return laplacian(request.polynomial)


def _directional_derivative(request: DirectionalDerivativeRequest) -> ScalarResult:
    return directional_derivative(request.polynomial, request.direction)


def _divergence(request: VectorFieldRequest) -> ScalarResult:
    return divergence(request.components)


def _curl(request: CurlRequest) -> VectorResult:
    return curl(request.components)


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[int, tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": list(variables),
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polynomial_field.scalar.gradient.compute",
        "Compute the gradient of a scalar field",
        "Compute the gradient vector of a multivariate polynomial scalar "
        "field using exact symbolic differentiation.",
        ScalarFieldRequest,
        VectorResult,
        _gradient,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "gradient_x2_y2",
                "Compute the gradient of x^2 + y^2; the canonical polynomial "
                "carries the complete ordered axis (x, y).",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                    )
                },
            ),
        ),
    ),
    _op(
        "polynomial_field.scalar.laplacian.compute",
        "Compute the Laplacian of a scalar field",
        "Compute the Laplacian (sum of second partial derivatives) of a "
        "multivariate polynomial scalar field.",
        ScalarFieldRequest,
        ScalarResult,
        _laplacian,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "laplacian_x2_y2",
                "Compute the Laplacian of x^2 + y^2; the canonical polynomial "
                "carries the complete ordered axis (x, y).",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                    )
                },
            ),
        ),
    ),
    _op(
        "polynomial_field.scalar.directional_derivative.compute",
        "Compute the directional derivative",
        "Compute the directional derivative of a scalar field along a "
        "direction vector using exact symbolic differentiation.",
        DirectionalDerivativeRequest,
        ScalarResult,
        _directional_derivative,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "directional_deriv_x2_y2",
                "Compute the directional derivative of x^2 + y^2 along the "
                "exact constant vector (1, 1); its length must match the axis.",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                    ),
                    "direction": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    _op(
        "polynomial_field.vector.divergence.compute",
        "Compute the divergence of a vector field",
        "Compute the divergence of a multivariate polynomial vector field "
        "using exact symbolic differentiation.",
        VectorFieldRequest,
        ScalarResult,
        _divergence,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "divergence_xy",
                "Compute the divergence of [x^2, y^2]; each component must "
                "use the same complete ordered axis (x, y).",
                {
                    "components": [
                        _polynomial(("x", "y"), (1, (2, 0))),
                        _polynomial(("x", "y"), (1, (0, 2))),
                    ],
                },
            ),
        ),
    ),
    _op(
        "polynomial_field.vector.curl.compute",
        "Compute the curl of a 3D vector field",
        "Compute the curl of a 3D multivariate polynomial vector field "
        "using exact symbolic differentiation.",
        CurlRequest,
        VectorResult,
        _curl,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "curl_constant_field",
                "Compute the curl of [y, 0, 0]; curl requires exactly three "
                "components on the ordered axis (x, y, z).",
                {
                    "components": [
                        _polynomial(("x", "y", "z"), (1, (0, 1, 0))),
                        _polynomial(("x", "y", "z")),
                        _polynomial(("x", "y", "z")),
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
