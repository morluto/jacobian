"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.moments_orthogonal._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "moments.hankel_matrix.compute",
        AdmissionDecision.KEEP,
        "exact Hankel matrix assembled from a bounded rational moment sequence",
    ),
    OperationAdmission(
        "moments.recurrence_coefficients.compute",
        AdmissionDecision.KEEP,
        "exact monic three-term recurrence coefficients from Gram-Schmidt orthogonalization",
    ),
    OperationAdmission(
        "moments.jacobi_matrix.compute",
        AdmissionDecision.KEEP,
        "exact symmetric tridiagonal Jacobi matrix from recurrence coefficients",
    ),
    OperationAdmission(
        "moments.christoffel_darboux.compute",
        AdmissionDecision.KEEP,
        "exact Christoffel-Darboux kernel by forward polynomial recurrence",
    ),
    OperationAdmission(
        "moments.gaussian_quadrature.compute",
        AdmissionDecision.KEEP,
        "Approximate Gaussian quadrature nodes and weights from Golub-Welsch "
        "IEEE-double eigenvalue decomposition (dyadic rational images of doubles, "
        "not exact algebraic numbers)",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
