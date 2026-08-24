"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.isomorphism._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.isomorphism.decide.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "graph.isomorphism.canonicalize.compute",
        AdmissionDecision.KEEP,
        "distinct exact canonical colored-graph value with a replayable source-to-canonical relabeling",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
