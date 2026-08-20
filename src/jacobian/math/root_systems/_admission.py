"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.root_systems._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "root_system.positive_roots.compute",
        AdmissionDecision.KEEP,
        "complete positive roots, highest root, and Coxeter number from a Cartan matrix",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
