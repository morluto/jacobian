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
        "distinct exact or explicitly bounded search outcome with material computational leverage; "
        "re-admitted for the request-scoped solver envelope: structural CNF budgets bound the input "
        "before Z3 runs, timeout, rlimit work, and max_memory ceilings bound the solve, and an "
        "incomplete result is a typed UNKNOWN that names the exhausted resource",
    ),
    OperationAdmission(
        "sat.refutation.check",
        AdmissionDecision.KEEP,
        "distinct source-bound certificate relation that cannot be established by SAT solving",
    ),
    OperationAdmission(
        "smt.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage; "
        "re-admitted for the bounded SMT-LIB envelope: ASCII bytes, nesting depth, compound terms, "
        "declared symbols, and numeral width (including indexed bit-vector values) are admitted "
        "before Z3 parses, request-scoped timeout, rlimit work, and max_memory ceilings bound the "
        "solve, and an incomplete result is a typed UNKNOWN that names the exhausted resource",
    ),
    OperationAdmission(
        "smt.unsat_core",
        AdmissionDecision.KEEP,
        "replayable source-bound contradiction certificate with distinct proof-extraction leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
