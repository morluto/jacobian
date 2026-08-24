"""Owner-local admission decision for oriented-matroid operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.oriented_matroids._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "oriented_matroid.chirotope.check",
        AdmissionDecision.KEEP,
        "exact bounded validity check with a source-bound first axiom obstruction",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
