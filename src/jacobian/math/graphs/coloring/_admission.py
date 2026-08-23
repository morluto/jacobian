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
        "graph.edge_coloring.check",
        AdmissionDecision.KEEP,
        "exact proper-edge-coloring validator over the canonical source-bound assignment value, the independent checker for edge-coloring producers",
    ),
    OperationAdmission(
        "graph.edge_coloring.k_decide",
        AdmissionDecision.KEEP,
        "exact bounded k-edge-colorability decision with a coloring witness, distinct from vertex k-colorability",
    ),
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
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
