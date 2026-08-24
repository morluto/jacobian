"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.discrepancy_theory._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "discrepancy.hard_constraint_round.compute",
        AdmissionDecision.KEEP,
        "distinct exact source-bound binary rounding that jointly preserves disjoint integral cardinality constraints and certifies the theorem-backed 4d monitored-column error bound",
    ),
    OperationAdmission(
        "discrepancy.theory.eval.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "discrepancy.theory.optimum.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
