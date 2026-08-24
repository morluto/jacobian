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
        "version-3 re-admission: kernel swapped from 2^n enumeration to one exact HiGHS "
        "MILP (scipy.optimize.milp, zero MIP gap) with a source-bound OPTIMAL result whose "
        "witness replays exactly and whose minimality is re-established by a proven-infeasible "
        "feasibility program; BUDGET_EXCEEDED and the new EXECUTION_FAILED outcomes carry no "
        "mathematical claim. Distinct bounded search outcome with materially wider leverage "
        "(64-element ground sets vs the retired 20-element scan).",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
