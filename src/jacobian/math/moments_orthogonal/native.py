"""Native domain functions over canonical moment-functional values."""

from __future__ import annotations

from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.moments_orthogonal.operations import (
    compute_christoffel_darboux,
    compute_gaussian_quadrature,
    compute_hankel_matrix,
    compute_jacobi_matrix,
    compute_orthogonal_polynomials,
    compute_recurrence,
    compute_shifted_hankel,
)
from jacobian.math.moments_orthogonal.values import (
    ChristoffelDarbouxKernel,
    GaussianQuadratureRule,
    HankelMomentMatrix,
    JacobiMatrix,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
    ThreeTermRecurrence,
)

__all__ = [
    "christoffel_darboux_kernel",
    "gaussian_quadrature_rule",
    "hankel_matrix",
    "jacobi_matrix",
    "orthogonal_polynomials",
    "recurrence_coefficients",
    "shifted_hankel_matrix",
]


def hankel_matrix(prefix: MomentFunctionalPrefix, order: int) -> HankelMomentMatrix:
    """Exact Hankel matrix H_order[i,j] = mu_(i+j) of one moment prefix."""
    return compute_hankel_matrix(HankelRequest(prefix=prefix, order=order))


def shifted_hankel_matrix(
    prefix: MomentFunctionalPrefix, order: int
) -> HankelMomentMatrix:
    """Exact shifted Hankel matrix H_order^(1)[i,j] = mu_(i+j+1)."""
    return compute_shifted_hankel(ShiftedHankelRequest(prefix=prefix, order=order))


def orthogonal_polynomials(
    prefix: MomentFunctionalPrefix, max_degree: int
) -> OrthogonalPolynomialFamily:
    """Exact monic orthogonal polynomial family p_0,...,p_max_degree."""
    return compute_orthogonal_polynomials(
        OrthogonalPolynomialRequest(prefix=prefix, max_degree=max_degree)
    )


def recurrence_coefficients(family: OrthogonalPolynomialFamily) -> ThreeTermRecurrence:
    """Exact three-term recurrence coefficients of one orthogonal family."""
    return compute_recurrence(RecurrenceRequest(family=family))


def christoffel_darboux_kernel(
    family: OrthogonalPolynomialFamily, degree: int
) -> ChristoffelDarbouxKernel:
    """Exact Christoffel-Darboux kernel K_degree(x,y) of one family."""
    return compute_christoffel_darboux(
        ChristoffelDarbouxRequest(family=family, degree=degree)
    )


def jacobi_matrix(family: OrthogonalPolynomialFamily) -> JacobiMatrix:
    """Exact finite tridiagonal Jacobi matrix of one orthogonal family."""
    return compute_jacobi_matrix(JacobiMatrixRequest(family=family))


def gaussian_quadrature_rule(
    prefix: MomentFunctionalPrefix, order: int
) -> GaussianQuadratureRule:
    """Exact Gaussian quadrature rule of one bounded moment prefix."""
    return compute_gaussian_quadrature(
        GaussianQuadratureRequest(prefix=prefix, order=order)
    )
