"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "petri_net.fire_transition.compute",
        AdmissionDecision.KEEP,
        "exact transition firing with typed marking semantics",
    ),
    OperationAdmission(
        "petri_net.reachability_graph.compute",
        AdmissionDecision.KEEP,
        "bounded reachability construction with an explicit typed completeness frontier after the #1978 contract repair",
    ),
)
