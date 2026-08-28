"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "algebraic_number.compare",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.roots.isolate",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
