"""Owner-local admission decisions for graph constructor operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.constructors._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.hypercube.construct",
        AdmissionDecision.KEEP,
        "distinct exact source-bound labelled graph constructor with canonical binary-string vertex order",
    ),
    OperationAdmission(
        "graph.keller.construct",
        AdmissionDecision.KEEP,
        "distinct exact source-bound labelled graph constructor with canonical base-4 word vertex order and Keller adjacency predicate",
    ),
    OperationAdmission(
        "graph.triangle_profile.compute",
        AdmissionDecision.KEEP,
        "distinct complete source-bound triangle profile retaining vertex-triple provenance for each triangle",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
