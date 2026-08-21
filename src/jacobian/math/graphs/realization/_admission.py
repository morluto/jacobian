"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.realization._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.realization.check.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.realization.construct.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.realization.is_graphical.compute",
        AdmissionDecision.DROP,
        "boolean projection already determined by graph.realization.construct.compute",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
