"""Moment-functional operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.analysis.orthogonal_polynomials._jacobi import (
    JacobiMatrixAdmissionError,
)
from jacobian.math.analysis.orthogonal_polynomials._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.analysis.orthogonal_polynomials.operations import (
    ChristoffelDarbouxAdmissionError,
    HankelMatrixAdmissionError,
    MomentsOrthogonalAdmissionError,
    christoffel_darboux_kernel,
    gaussian_quadrature_rule,
    hankel_matrix,
    jacobi_matrix,
    orthogonal_polynomials,
    recurrence_coefficients,
    shifted_hankel_matrix,
)
from jacobian.math.analysis.orthogonal_polynomials.values import (
    ChristoffelDarbouxKernel,
    GaussianQuadratureRule,
    HankelMomentMatrix,
    JacobiMatrix,
    OrthogonalPolynomialFamily,
    ThreeTermRecurrence,
)


def compute_hankel_matrix(request: HankelRequest) -> HankelMomentMatrix:
    """Project one wire request into the canonical Hankel operation."""
    try:
        return hankel_matrix(request.prefix, request.order)
    except HankelMatrixAdmissionError as exc:
        location = (
            ("prefix", "moments") if exc.reason != "order_out_of_range" else ("order",)
        )
        raise OperationDomainValidationError(
            location=location,
            code=f"moment_functional.hankel.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_shifted_hankel(request: ShiftedHankelRequest) -> HankelMomentMatrix:
    """Project one wire request into the canonical shifted Hankel operation."""
    try:
        return shifted_hankel_matrix(request.prefix, request.order)
    except HankelMatrixAdmissionError as exc:
        location = (
            ("prefix", "moments") if exc.reason != "order_out_of_range" else ("order",)
        )
        raise OperationDomainValidationError(
            location=location,
            code=f"moment_functional.shifted_hankel.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_orthogonal_polynomials(
    request: OrthogonalPolynomialRequest,
) -> OrthogonalPolynomialFamily:
    """Project one wire request into the canonical family operation."""
    try:
        return orthogonal_polynomials(request.prefix, request.max_degree)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prefix", "max_degree"),
            code="moments_orthogonal.family_not_admitted",
            message=str(exc),
        ) from exc


def compute_recurrence(request: RecurrenceRequest) -> ThreeTermRecurrence:
    """Project one wire request into the canonical recurrence operation."""
    try:
        return recurrence_coefficients(request.family)
    except MomentsOrthogonalAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family",),
            code=f"moments_orthogonal.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_christoffel_darboux(
    request: ChristoffelDarbouxRequest,
) -> ChristoffelDarbouxKernel:
    """Project one wire request into the canonical kernel operation."""
    try:
        return christoffel_darboux_kernel(request.family, request.degree)
    except ChristoffelDarbouxAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family", "degree"),
            code=f"moments_orthogonal.christoffel_darboux.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_jacobi_matrix(request: JacobiMatrixRequest) -> JacobiMatrix:
    """Project one wire request into the canonical Jacobi operation."""
    try:
        return jacobi_matrix(request.family)
    except JacobiMatrixAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("family",),
            code=f"moments_orthogonal.jacobi_matrix.{exc.reason}",
            message=str(exc),
        ) from exc


def compute_gaussian_quadrature(
    request: GaussianQuadratureRequest,
) -> GaussianQuadratureRule:
    """Project one wire request into the canonical quadrature operation."""
    try:
        return gaussian_quadrature_rule(request.prefix, request.order)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prefix", "order"),
            code="moments_orthogonal.quadrature_not_admitted",
            message=str(exc),
        ) from exc


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
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_MOMENTS = [
    {"num": "2", "den": "1"},
    {"num": "0", "den": "1"},
    {"num": "2", "den": "3"},
    {"num": "0", "den": "1"},
    {"num": "2", "den": "5"},
    {"num": "0", "den": "1"},
    {"num": "2", "den": "7"},
    {"num": "0", "den": "1"},
    {"num": "2", "den": "9"},
]

_TOY_MOMENTS = [
    {"num": "1", "den": "1"},
    {"num": "0", "den": "1"},
    {"num": "1", "den": "3"},
    {"num": "0", "den": "1"},
    {"num": "1", "den": "5"},
    {"num": "0", "den": "1"},
    {"num": "1", "den": "7"},
]


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "moment_functional.hankel.compute",
        "Compute Hankel moment matrix",
        "Compute the exact Hankel matrix H_r[i,j] = mu_(i+j) from a bounded "
        "rational moment prefix, with determinant and rank.",
        HankelRequest,
        HankelMomentMatrix,
        compute_hankel_matrix,
        "moments",
        "hankel",
        "exact",
        examples=(
            example(
                "uniform_hankel",
                "Hankel matrix for the uniform measure on [-1,1]: "
                "mu_k = 2/(k+1) for even k and mu_k = 0 for odd k. The "
                "prefix must hold at least 2*order+1 moments scaled so the "
                "exact determinant stays canonical.",
                {"prefix": {"moments": _MOMENTS[:5], "variable": "x"}, "order": 2},
            ),
        ),
    ),
    _op(
        "moment_functional.shifted_hankel.compute",
        "Compute shifted Hankel moment matrix",
        "Compute the exact shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1).",
        ShiftedHankelRequest,
        HankelMomentMatrix,
        compute_shifted_hankel,
        "moments",
        "hankel",
        "exact",
        examples=(
            example(
                "shifted_hankel_uniform",
                "Shifted Hankel matrix for the uniform measure on [-1,1]: "
                "odd moments vanish, so H^(1) reads mu_1..mu_(2*order+1). "
                "The prefix must hold at least 2*order+2 moments within the "
                "determinant height bound.",
                {"prefix": {"moments": _MOMENTS[:6], "variable": "x"}, "order": 2},
            ),
        ),
    ),
    _op(
        "moment_functional.orthogonal_polynomials.compute",
        "Compute monic orthogonal polynomials from moments",
        "Compute the exact monic orthogonal polynomial family p_0,...,p_n "
        "from a bounded rational moment prefix using exact Gram-Schmidt.",
        OrthogonalPolynomialRequest,
        OrthogonalPolynomialFamily,
        compute_orthogonal_polynomials,
        "moments",
        "orthogonal-polynomials",
        "exact",
        examples=(
            example(
                "legendre_from_uniform",
                "Monic Legendre-like polynomials from the uniform moments "
                "on [-1,1] (odd moments vanish). The prefix needs at least "
                "2*max_degree+1 moments that stay quasi-definite through "
                "max_degree and meet the Gram-Schmidt height bound.",
                {"prefix": {"moments": _MOMENTS[:7], "variable": "x"}, "max_degree": 3},
            ),
        ),
    ),
    _op(
        "orthogonal_polynomial.recurrence.compute",
        "Compute three-term recurrence coefficients",
        "Compute the exact three-term recurrence alpha_k, beta_k from an "
        "orthogonal polynomial family.",
        RecurrenceRequest,
        ThreeTermRecurrence,
        compute_recurrence,
        "moments",
        "recurrence",
        "exact",
        examples=(
            example(
                "recurrence_from_legendre",
                "Recurrence of a monic Legendre-like family. The family "
                "must be quasi-definite with a nonzero squared norm for "
                "every supplied polynomial.",
                {
                    "family": {
                        "polynomials": [
                            {
                                "degree": 0,
                                "coefficients": [{"num": "1", "den": "1"}],
                                "squared_norm": {"num": "2", "den": "1"},
                            },
                            {
                                "degree": 1,
                                "coefficients": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                                "squared_norm": {"num": "2", "den": "3"},
                            },
                        ],
                        "variable": "x",
                        "is_quasi_definite": True,
                        "is_positive_definite": True,
                    },
                },
            ),
        ),
    ),
    _op(
        "orthogonal_polynomial.christoffel_darboux.compute",
        "Compute Christoffel-Darboux kernel",
        "Compute the exact Christoffel-Darboux kernel K_m(x,y) from an "
        "orthogonal polynomial family.",
        ChristoffelDarbouxRequest,
        ChristoffelDarbouxKernel,
        compute_christoffel_darboux,
        "moments",
        "christoffel-darboux",
        "exact",
        examples=(
            example(
                "cd_kernel_degree_0",
                "Christoffel-Darboux kernel of degree 0. The requested "
                "degree must stay below the family size, and every squared "
                "norm through that degree must be nonzero.",
                {
                    "family": {
                        "polynomials": [
                            {
                                "degree": 0,
                                "coefficients": [{"num": "1", "den": "1"}],
                                "squared_norm": {"num": "2", "den": "1"},
                            },
                            {
                                "degree": 1,
                                "coefficients": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                                "squared_norm": {"num": "2", "den": "3"},
                            },
                        ],
                        "variable": "x",
                        "is_quasi_definite": True,
                        "is_positive_definite": True,
                    },
                    "degree": 0,
                },
            ),
        ),
    ),
    _op(
        "orthogonal_polynomial.jacobi_matrix.compute",
        "Compute the finite Jacobi matrix",
        "Compute the exact finite tridiagonal Jacobi matrix from an "
        "orthogonal polynomial family.",
        JacobiMatrixRequest,
        JacobiMatrix,
        compute_jacobi_matrix,
        "moments",
        "jacobi-matrix",
        "exact",
        examples=(
            example(
                "jacobi_matrix_legendre",
                "Jacobi matrix of a monic Legendre-like family. Adjacent "
                "squared norms feeding an emitted ratio must be nonzero, "
                "and derived recurrence entries must stay canonical.",
                {
                    "family": {
                        "polynomials": [
                            {
                                "degree": 0,
                                "coefficients": [{"num": "1", "den": "1"}],
                                "squared_norm": {"num": "2", "den": "1"},
                            },
                            {
                                "degree": 1,
                                "coefficients": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                                "squared_norm": {"num": "2", "den": "3"},
                            },
                        ],
                        "variable": "x",
                        "is_quasi_definite": True,
                        "is_positive_definite": True,
                    },
                },
            ),
        ),
    ),
    _op(
        "moment_functional.gaussian_quadrature.compute",
        "Compute exact Gaussian quadrature rule",
        "Compute an exact Gaussian quadrature rule from a bounded moment "
        "prefix whose degree-n orthogonal polynomial splits over QQ: 2n "
        "moments, n distinct rational nodes, positive weights, exactness "
        "through degree 2n-1.",
        GaussianQuadratureRequest,
        GaussianQuadratureRule,
        compute_gaussian_quadrature,
        "moments",
        "gaussian-quadrature",
        "exact",
        examples=(
            example(
                "gaussian_quadrature_rational_nodes",
                (
                    "Gaussian quadrature for weight 7 at +-1 and 5 at +-2: "
                    "nodes +-3/2, weight 12. The prefix needs at least "
                    "2*order canonical moments whose degree-order "
                    "orthogonal polynomial splits over QQ into distinct "
                    "factors with positive weights."
                ),
                {
                    "prefix": {
                        "moments": [
                            {"num": "24", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "54", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "174", "den": "1"},
                        ],
                        "variable": "x",
                    },
                    "order": 2,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
