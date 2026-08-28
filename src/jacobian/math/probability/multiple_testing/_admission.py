"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.probability.multiple_testing._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "probability.bh_step_up.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "probability.fdp.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of two caller-supplied finite sets",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
