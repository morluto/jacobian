"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.galois._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.galois.factor_mod_p.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.galois.frobenius_cycle.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.galois_group.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.solvable_by_radicals.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
