"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.analysis.boolean._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "boolean.fourier.walsh_transform.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
