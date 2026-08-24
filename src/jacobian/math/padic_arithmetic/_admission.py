"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.padic_arithmetic._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.padic.hensel_root.compute",
        AdmissionDecision.KEEP,
        "exact Hensel root lifting from mod p to mod p^k with simple-root certification",
    ),
    OperationAdmission(
        "number_theory.padic.roots.compute",
        AdmissionDecision.KEEP,
        "exact p-adic root finding via Hensel lifting with simple/multiple root classification",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
