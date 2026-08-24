"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.morphisms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.cycle.fixed_length.decide",
        AdmissionDecision.KEEP,
        "exact bounded witness-producing search for a fixed-length simple cycle, distinct from girth and Hamiltonicity",
    ),
    OperationAdmission(
        "graph.subgraph_pattern.find",
        AdmissionDecision.KEEP,
        "exact bounded injective edge-preserving subgraph-monomorphism search with material leverage over bespoke enumeration",
    ),
    OperationAdmission(
        "graph.homomorphism.check",
        AdmissionDecision.KEEP,
        "complete canonical graph-map check with a reusable source-bound positive value or exact first edge-image obstruction",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
