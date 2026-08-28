"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.logic.automata.petri_nets._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "petri_net.fire_transition.compute",
        AdmissionDecision.KEEP,
        "exact transition firing with typed marking semantics",
    ),
    OperationAdmission(
        "petri_net.reachability_graph.compute",
        AdmissionDecision.KEEP,
        "aggregate-bounded reachability with exact frontier and marking-envelope escape witnesses",
    ),
    OperationAdmission(
        "petri_net.enabled_transitions.compute",
        AdmissionDecision.KEEP,
        "exact enabled-transition indices for a bounded marking",
    ),
    OperationAdmission(
        "petri_net.incidence_matrix.compute",
        AdmissionDecision.KEEP,
        "exact incidence matrix of a bounded Petri net",
    ),
    OperationAdmission(
        "petri_net.siphon_trap.check",
        AdmissionDecision.KEEP,
        "exact minimal siphon and trap witnesses under bounded place enumeration",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
