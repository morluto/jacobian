"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.additive_combinatorics._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "additive.direct_sum_predicate.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.energy.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
    OperationAdmission(
        "additive.representation_profile.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "additive.sumset_cardinality.compute",
        AdmissionDecision.DROP,
        "cheap deterministic projection of additive.representation_profile.compute",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
