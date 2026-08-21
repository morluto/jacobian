"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.cycle_pattern._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.cycle.fixed_length.decide",
        AdmissionDecision.KEEP,
        "exhaustive bounded search for a simple k-cycle witness with typed semantics",
    ),
    OperationAdmission(
        "graph.subgraph.pattern.find",
        AdmissionDecision.KEEP,
        "exhaustive bounded subgraph-isomorphism search with typed embedding witness",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
