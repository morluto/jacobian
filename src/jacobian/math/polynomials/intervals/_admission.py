"""Owner-local admission decision for polynomial box enclosure."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.intervals._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.box.enclosure.compute",
        AdmissionDecision.KEEP,
        "reusable exact source-bound complete-box enclosure with a distinct discovery intent and material certification leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
