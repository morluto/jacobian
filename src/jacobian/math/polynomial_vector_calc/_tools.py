"""Polynomial vector calculus operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomial_vector_calc._models import (
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
)
from jacobian.math.polynomial_vector_calc._operations import (
    compute_curl,
    compute_directional_derivative,
    compute_divergence,
    compute_gradient,
    compute_laplacian,
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
        "polynomial_field.scalar.gradient.compute",
        "Compute the gradient of a scalar field",
        "Compute the gradient vector of a multivariate polynomial scalar "
        "field using exact symbolic differentiation.",
        ScalarFieldRequest,
        VectorResult,
        compute_gradient,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "gradient_x2_y2",
                "Gradient of x^2 + y^2 in (x, y).",
                {"variables": ["x", "y"], "polynomial": "x**2 + y**2"},
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
        compute_laplacian,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "laplacian_x2_y2",
                "Laplacian of x^2 + y^2 in (x, y).",
                {"variables": ["x", "y"], "polynomial": "x**2 + y**2"},
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
        compute_directional_derivative,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "directional_deriv_x2_y2",
                "Directional derivative of x^2 + y^2 along (1, 1).",
                {
                    "variables": ["x", "y"],
                    "polynomial": "x**2 + y**2",
                    "direction": ["1", "1"],
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
        compute_divergence,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "divergence_xy",
                "Divergence of [x^2, y^2] in (x, y).",
                {
                    "variables": ["x", "y"],
                    "components": ["x**2", "y**2"],
                },
            ),
        ),
    ),
    _op(
        "polynomial_field.vector.curl.compute",
        "Compute the curl of a 3D vector field",
        "Compute the curl of a 3D multivariate polynomial vector field "
        "using exact symbolic differentiation.",
        VectorFieldRequest,
        VectorResult,
        compute_curl,
        "polynomial",
        "vector-calculus",
        "exact",
        examples=(
            example(
                "curl_constant_field",
                "Curl of [y, 0, 0] in (x, y, z).",
                {
                    "variables": ["x", "y", "z"],
                    "components": ["y", "0", "0"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
