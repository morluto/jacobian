"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry.inversion._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.inversion.circle_inversion.compute",
        AdmissionDecision.KEEP,
        "exact rational circle inversion with typed power semantics and center rejection",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
