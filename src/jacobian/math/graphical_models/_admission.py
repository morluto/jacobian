"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphical_models._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graphical_model.d_separation.compute",
        AdmissionDecision.KEEP,
        "exact d-separation verdict for a bounded directed acyclic graphical model",
    ),
    OperationAdmission(
        "graphical_model.factor.marginalize",
        AdmissionDecision.KEEP,
        "exact bounded factor marginalization",
    ),
    OperationAdmission(
        "graphical_model.factor.multiply",
        AdmissionDecision.KEEP,
        "exact bounded factor multiplication",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
