"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
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
