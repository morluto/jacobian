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
    OperationAdmission(
        "quadratic_form.representation_numbers.compute",
        AdmissionDecision.KEEP,
        "exact representation numbers r(n) by brute-force enumeration",
    ),
    OperationAdmission(
        "quadratic_form.theta_series_prefix.compute",
        AdmissionDecision.KEEP,
        "exact theta series prefix coefficients",
    ),
    OperationAdmission(
        "quadratic_form.scale.compute",
        AdmissionDecision.KEEP,
        "exact integer scaling of a quadratic form",
    ),
    OperationAdmission(
        "quadratic_form.direct_sum.compute",
        AdmissionDecision.KEEP,
        "exact block diagonal direct sum of two quadratic forms",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
