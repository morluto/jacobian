"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.group._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "group.conjugacy_classes.compute",
        AdmissionDecision.KEEP,
        "exact conjugacy class computation with typed class representatives for bounded permutation groups",
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
        "exact subgroup lattice enumeration with typed generator records for bounded permutation groups",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
