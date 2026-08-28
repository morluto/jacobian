"""Exact moment-functional and orthogonal-polynomial algebra."""

from jacobian.math.analysis.orthogonal_polynomials.operations import (
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
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
    OrthogonalPolynomialTerm,
    QuadratureNode,
    ThreeTermRecurrence,
)

# The native surface accepts canonical domain values directly. The
# compute_* wire-envelope handlers stay private to the MCP declarations
# (they require operation-specific request envelopes) and are not part
# of the supported Python API.
__all__ = [
    "ChristoffelDarbouxKernel",
    "GaussianQuadratureRule",
    "HankelMomentMatrix",
    "JacobiMatrix",
    "MomentFunctionalPrefix",
    "OrthogonalPolynomialFamily",
    "OrthogonalPolynomialTerm",
    "QuadratureNode",
    "ThreeTermRecurrence",
    "christoffel_darboux_kernel",
    "gaussian_quadrature_rule",
    "hankel_matrix",
    "jacobi_matrix",
    "orthogonal_polynomials",
    "recurrence_coefficients",
    "shifted_hankel_matrix",
]
