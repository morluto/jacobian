"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "cubical.f_vector.compute",
        AdmissionDecision.KEEP,
        "exact f-vector and Euler characteristic of a finite cubical complex",
    ),
    OperationAdmission(
        "cubical.face_closure.compute",
        AdmissionDecision.KEEP,
        "complete face closure with cells by dimension for a cubical complex",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
