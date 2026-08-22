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
        "moment_functional.hankel.compute",
        AdmissionDecision.KEEP,
        "exact Hankel matrix from a bounded rational moment prefix",
    ),
    OperationAdmission(
        "moment_functional.shifted_hankel.compute",
        AdmissionDecision.KEEP,
        "exact shifted Hankel matrix from a bounded rational moment prefix",
    ),
    OperationAdmission(
        "moment_functional.orthogonal_polynomials.compute",
        AdmissionDecision.KEEP,
        "exact monic orthogonal polynomial family from moments via Gram-Schmidt",
    ),
    OperationAdmission(
        "orthogonal_polynomial.recurrence.compute",
        AdmissionDecision.KEEP,
        "exact three-term recurrence coefficients from an orthogonal family",
    ),
    OperationAdmission(
        "orthogonal_polynomial.christoffel_darboux.compute",
        AdmissionDecision.KEEP,
        "exact Christoffel-Darboux kernel from an orthogonal family",
    ),
    OperationAdmission(
        "orthogonal_polynomial.jacobi_matrix.compute",
        AdmissionDecision.KEEP,
        "exact finite Jacobi matrix from an orthogonal family",
    ),
    OperationAdmission(
        "moment_functional.gaussian_quadrature.compute",
        AdmissionDecision.KEEP,
        "exact Gaussian quadrature rule with exactness through degree 2n-1",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
