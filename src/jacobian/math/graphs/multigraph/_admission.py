"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.multigraph._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.multigraph.cycle_multicover.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with"
        " material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.multigraph.eulerian_cycles.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with"
        " material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.multigraph.flow.finite_abelian.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with"
        " material computational or reliability leverage",
    ),
    OperationAdmission(
        "graph.multigraph.flow.finite_abelian.find",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with"
        " material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
