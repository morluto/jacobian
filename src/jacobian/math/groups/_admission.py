"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.groups._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "group.conjugacy_classes.compute",
        AdmissionDecision.KEEP,
        "exact conjugacy-class partition of a permutation group via SymPy, a fundamental group invariant",
    ),
    OperationAdmission(
        "group.stabilizer.compute",
        AdmissionDecision.KEEP,
        "exact point-stabilizer generators via SymPy, the orbit-stabilizer complement to group.orbit.compute and group.order.compute",
    ),
    OperationAdmission(
        "group.element_order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "group.orbit.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "group.order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "group.subgroup_lattice.compute",
        AdmissionDecision.KEEP,
        "exact complete subgroup lattice enumeration via SymPy bounded to groups of order 64",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
