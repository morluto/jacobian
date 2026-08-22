"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.approximation_theory._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "approximation.lagrange.basis.compute",
        AdmissionDecision.KEEP,
        "exact Lagrange basis polynomials and barycentric weights over QQ",
    ),
    OperationAdmission(
        "approximation.lagrange.interpolate.compute",
        AdmissionDecision.KEEP,
        "exact Lagrange interpolation polynomial over QQ",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
