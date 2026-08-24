"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.optimization._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "optimization.linear.rational_optimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact source-bound standard-form LP outcome with replayable optimality, Farkas, or recession witnesses",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
