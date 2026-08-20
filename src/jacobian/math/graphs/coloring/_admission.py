"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.coloring._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.coloring.k_colorability.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.independent_set.maximal.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.independent_set.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
