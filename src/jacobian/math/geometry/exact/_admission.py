"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.points.distance_graph.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "geometry.points.distance_profile.compute",
        AdmissionDecision.KEEP,
        "one complete exact multiplicity profile of the pairwise-distance multiset",
    ),
)
