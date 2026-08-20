"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.majorization._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "majorization.check.compute",
        AdmissionDecision.KEEP,
        "exact majorization check with prefix-sum profile",
    ),
    OperationAdmission(
        "majorization.weak_check.compute",
        AdmissionDecision.KEEP,
        "exact weak majorization check with sub/super direction",
    ),
    OperationAdmission(
        "majorization.t_transform.compute",
        AdmissionDecision.KEEP,
        "exact T-transform sequence with doubly stochastic composition",
    ),
    OperationAdmission(
        "majorization.doubly_stochastic.check",
        AdmissionDecision.KEEP,
        "exact doubly stochastic matrix verification",
    ),
    OperationAdmission(
        "majorization.birkhoff_decomposition.compute",
        AdmissionDecision.KEEP,
        "exact Birkhoff-von Neumann decomposition into permutation matrices",
    ),
    OperationAdmission(
        "majorization.schur_horn.check",
        AdmissionDecision.KEEP,
        "exact Schur-Horn feasibility check",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
