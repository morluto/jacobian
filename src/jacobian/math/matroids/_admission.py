"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matroids._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matroid.rank.compute",
        AdmissionDecision.KEEP,
        "exact matroid rank via Gaussian elimination over a prime field",
    ),
    OperationAdmission(
        "matroid.closure.compute",
        AdmissionDecision.KEEP,
        "exact matroid closure (flat) computation via column span over a prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
