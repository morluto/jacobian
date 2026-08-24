"""Owner-local admission for rational-function operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.rational_functions._tools import TOOLS

ADMISSIONS = (
    OperationAdmission(
        "rational_function.hermite_reduction.compute",
        AdmissionDecision.KEEP,
        "canonical quotient by exact rational derivatives with a complete primitive decision",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["REGISTRATION"]
