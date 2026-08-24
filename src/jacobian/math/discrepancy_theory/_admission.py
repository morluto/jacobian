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
        "version-3 re-admission against the direct parent, which already solved "
        "exactly with Z3 Optimize over binary color bits at a 64-element ceiling: "
        "this candidate replaces that single-solver kernel with a bounded "
        "scipy.optimize.milp (HiGHS) incumbent search plus a mandatory exact Z3 "
        "pseudo-boolean feasibility proof at D-1 before any OPTIMAL claim, so the "
        "published lower bound stays exact while the incumbent search scales. The "
        "OPTIMAL witness replays exactly against its source system; BUDGET_EXCEEDED "
        "and EXECUTION_FAILED carry no mathematical claim. Distinct bounded search "
        "outcome retaining the parent's 64-element ground-set leverage.",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
