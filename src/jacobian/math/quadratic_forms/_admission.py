"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.quadratic_forms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "quadratic_form.evaluate.compute",
        AdmissionDecision.KEEP,
        "exact integer evaluation q(x) = x^T A x for an integral quadratic form",
    ),
    OperationAdmission(
        "quadratic_form.discriminant.compute",
        AdmissionDecision.KEEP,
        "exact determinant of the symmetric matrix via SymPy",
    ),
    OperationAdmission(
        "quadratic_form.signature.compute",
        AdmissionDecision.KEEP,
        "exact inertia and definiteness classification via SymPy eigenvalues",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
