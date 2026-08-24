"""Owner-local admission for rational box-union operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry.boxes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.box_union.volume.compute",
        AdmissionDecision.KEEP,
        "one exact source-bound finite-union measure with a complete bounded intersection ledger",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
