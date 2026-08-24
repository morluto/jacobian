"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.tree_automata._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "tree_automaton.states.reachable.compute",
        AdmissionDecision.KEEP,
        "exact least-fixed-point state reachability with canonical bounded ground-tree witnesses",
    ),
    OperationAdmission(
        "tree_automaton.accepted_tree_count.compute",
        AdmissionDecision.KEEP,
        "exact bounded accepted-tree enumeration for a deterministic bottom-up tree automaton",
    ),
    OperationAdmission(
        "tree_automaton.run.compute",
        AdmissionDecision.KEEP,
        "typed bottom-up tree automaton run with a complete accepted or rejected verdict",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
