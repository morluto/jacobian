"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.flow._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.cut.minimum_st.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.flow.maximum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "graph.menger.edge_disjoint.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "network.min_cost_flow.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
