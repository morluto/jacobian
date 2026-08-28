"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.matroids._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matroid.closure.compute",
        AdmissionDecision.KEEP,
        "exact matroid closure (flat) via the shared prime-field column-rank kernel",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
