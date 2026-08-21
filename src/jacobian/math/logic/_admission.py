"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.logic._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "lean.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "sat.assignment.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "sat.cnf.canonicalize",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "sat.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "smt.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
