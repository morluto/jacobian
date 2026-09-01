"""Polynomial vector calculus operation declarations."""

from typing import Any

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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial_field.scalar.gradient.compute",
        title="Compute the gradient of a scalar field",
        description="Compute the gradient vector of a multivariate polynomial scalar "
        "field using exact symbolic differentiation.",
        request_type=ScalarFieldRequest,
        result_type=VectorResult,
        run=_gradient,
        tags=("polynomial", "vector-calculus", "exact"),
        examples=(
            OperationExample(
                name="gradient_x2_y2",
                description="Compute the gradient of x^2 + y^2; the canonical polynomial "
                "carries the complete ordered axis (x, y).",
                input={
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                    )
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_field.scalar.laplacian.compute",
        title="Compute the Laplacian of a scalar field",
        description="Compute the Laplacian (sum of second partial derivatives) of a "
        "multivariate polynomial scalar field.",
        request_type=ScalarFieldRequest,
        result_type=ScalarResult,
        run=_laplacian,
        tags=("polynomial", "vector-calculus", "exact"),
        examples=(
            OperationExample(
                name="laplacian_x2_y2",
                description="Compute the Laplacian of x^2 + y^2; the canonical polynomial "
                "carries the complete ordered axis (x, y).",
                input={
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                    )
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_field.scalar.directional_derivative.compute",
        title="Compute the directional derivative",
        description="Compute the directional derivative of a scalar field along a "
        "direction vector using exact symbolic differentiation.",
        request_type=DirectionalDerivativeRequest,
        result_type=ScalarResult,
        run=_directional_derivative,
        tags=("polynomial", "vector-calculus", "exact"),
        examples=(
            OperationExample(
                name="directional_deriv_x2_y2",
                description="Compute the directional derivative of x^2 + y^2 along the "
                "exact constant vector (1, 1); its length must match the axis.",
                input={
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
    MathTool(
        operation_id="polynomial_field.vector.divergence.compute",
        title="Compute the divergence of a vector field",
        description="Compute the divergence of a multivariate polynomial vector field "
        "using exact symbolic differentiation.",
        request_type=VectorFieldRequest,
        result_type=ScalarResult,
        run=_divergence,
        tags=("polynomial", "vector-calculus", "exact"),
        examples=(
            OperationExample(
                name="divergence_xy",
                description="Compute the divergence of [x^2, y^2]; each component must "
                "use the same complete ordered axis (x, y).",
                input={
                    "components": [
                        _polynomial(("x", "y"), (1, (2, 0))),
                        _polynomial(("x", "y"), (1, (0, 2))),
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_field.vector.curl.compute",
        title="Compute the curl of a 3D vector field",
        description="Compute the curl of a 3D multivariate polynomial vector field "
        "using exact symbolic differentiation.",
        request_type=CurlRequest,
        result_type=VectorResult,
        run=_curl,
        tags=("polynomial", "vector-calculus", "exact"),
        examples=(
            OperationExample(
                name="curl_constant_field",
                description="Compute the curl of [y, 0, 0]; curl requires exactly three "
                "components on the ordered axis (x, y, z).",
                input={
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
