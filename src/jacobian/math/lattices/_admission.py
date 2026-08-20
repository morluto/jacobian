"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.lattices._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "lattice.basis.reduce",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "lattice.hermite_normal_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
