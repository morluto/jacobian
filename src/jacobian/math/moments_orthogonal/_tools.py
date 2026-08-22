"""Moments and orthogonal polynomials operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    ChristoffelDarbouxResult,
    GaussianQuadratureRequest,
    GaussianQuadratureResult,
    HankelMatrixRequest,
    HankelMatrixResult,
    JacobiMatrixRequest,
    JacobiMatrixResult,
    RecurrenceCoefficientsRequest,
    RecurrenceCoefficientsResult,
)
from jacobian.math.moments_orthogonal._operations import (
    compute_christoffel_darboux,
    compute_gaussian_quadrature,
    compute_hankel_matrix,
    compute_jacobi_matrix,
    compute_recurrence_coefficients,
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
    examples: tuple[OperationExample, ...],
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_MOMENTS = [
    {"num": "1", "den": "1"},
    {"num": "1", "den": "2"},
    {"num": "1", "den": "3"},
    {"num": "1", "den": "4"},
    {"num": "1", "den": "5"},
    {"num": "1", "den": "6"},
    {"num": "1", "den": "7"},
]

_ALPHA = [
    {"num": "0", "den": "1"},
    {"num": "1", "den": "3"},
]

_BETA = [
    {"num": "2", "den": "1"},
    {"num": "1", "den": "9"},
    {"num": "4", "den": "45"},
]

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "moments.hankel_matrix.compute",
        "Compute the Hankel matrix from a moment sequence",
        "Build the exact Hankel matrix H[i][j] = mu_{i+j} from a bounded "
        "sequence of exact rational moments.",
        HankelMatrixRequest,
        HankelMatrixResult,
        compute_hankel_matrix,
        "linear-algebra",
        "moments",
        "hankel",
        "exact",
        examples=(
            example(
                "harmonic_moments",
                "Build the 4x4 Hankel matrix from the first 7 harmonic "
                "moments mu_k = 1/(k+1).",
                {"moments": _MOMENTS},
            ),
        ),
    ),
    _op(
        "moments.recurrence_coefficients.compute",
        "Compute monic orthogonal polynomial recurrence coefficients",
        "Compute exact three-term recurrence coefficients (alpha, beta) for "
        "the monic orthogonal polynomial family defined by a moment sequence, "
        "via exact Gram-Schmidt orthogonalization.",
        RecurrenceCoefficientsRequest,
        RecurrenceCoefficientsResult,
        compute_recurrence_coefficients,
        "orthogonal-polynomials",
        "moments",
        "recurrence",
        "exact",
        examples=(
            example(
                "harmonic_recurrence",
                "Compute recurrence coefficients from the first 7 harmonic "
                "moments mu_k = 1/(k+1).",
                {"moments": _MOMENTS},
            ),
        ),
    ),
    _op(
        "moments.jacobi_matrix.compute",
        "Assemble the Jacobi matrix from recurrence coefficients",
        "Assemble the symmetric tridiagonal Jacobi matrix from monic "
        "three-term recurrence coefficients, returning the rational diagonal "
        "and the rational squared subdiagonal entries.",
        JacobiMatrixRequest,
        JacobiMatrixResult,
        compute_jacobi_matrix,
        "linear-algebra",
        "orthogonal-polynomials",
        "jacobi-matrix",
        "exact",
        examples=(
            example(
                "three_point_jacobi",
                "Assemble the Jacobi matrix from three recurrence coefficients.",
                {"coefficients": {"alpha": _ALPHA, "beta": _BETA}},
            ),
        ),
    ),
    _op(
        "moments.christoffel_darboux.compute",
        "Compute the Christoffel-Darboux kernel",
        "Evaluate the Christoffel-Darboux kernel K_n(x, y) by forward "
        "recurrence of the monic orthogonal polynomials at the exact rational "
        "points x and y.",
        ChristoffelDarbouxRequest,
        ChristoffelDarbouxResult,
        compute_christoffel_darboux,
        "orthogonal-polynomials",
        "christoffel-darboux",
        "exact",
        examples=(
            example(
                "cd_at_1_1",
                "Evaluate the Christoffel-Darboux kernel at x=y=1.",
                {
                    "coefficients": {"alpha": _ALPHA, "beta": _BETA},
                    "x": {"num": "1", "den": "1"},
                    "y": {"num": "1", "den": "1"},
                },
            ),
        ),
    ),
    _op(
        "moments.gaussian_quadrature.compute",
        "Compute approximate Gaussian quadrature nodes and weights",
        "Compute approximate Gaussian quadrature nodes and weights from the "
        "symmetric tridiagonal Jacobi matrix via the Golub-Welsch algorithm "
        "(IEEE-double eigenvalue decomposition). Nodes are generally irrational "
        "(e.g. ±sqrt(2)) and returned as dyadic-rational images of doubles, "
        "explicitly approximate.",
        GaussianQuadratureRequest,
        GaussianQuadratureResult,
        compute_gaussian_quadrature,
        "numerical-integration",
        "gaussian-quadrature",
        "golub-welsch",
        "approximate",
        examples=(
            example(
                "two_point_gauss",
                "Compute 2-point Gaussian quadrature from 2 recurrence "
                "coefficients.",
                {
                    "coefficients": {
                        "alpha": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        "beta": [
                            {"num": "2", "den": "1"},
                            {"num": "1", "den": "3"},
                            {"num": "4", "den": "45"},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
