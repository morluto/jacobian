"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.analysis._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "analysis.real_function.point_enclosure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "interval.compute.enclosure",
        AdmissionDecision.KEEP,
        "distinct rigorous bounded enclosure for composable user-defined expressions",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
