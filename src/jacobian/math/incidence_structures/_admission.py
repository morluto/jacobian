"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.incidence_structures._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "incidence.matrix.compute",
        AdmissionDecision.KEEP,
        "exact labelled 0/1 incidence matrix with point rows and block columns",
    ),
    OperationAdmission(
        "incidence.degree_profile.compute",
        AdmissionDecision.KEEP,
        "per-point and per-block degree profiles with total incidence count",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
