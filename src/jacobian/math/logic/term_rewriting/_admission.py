"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.logic.term_rewriting._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "term_rewriting.critical_pairs.compute",
        AdmissionDecision.KEEP,
        "complete bounded first-order overlap family with replayable peak reducts; "
        "source, overlap work, output, and replay are preflight-bounded",
    ),
    OperationAdmission(
        "term_rewriting.matching.compute",
        AdmissionDecision.KEEP,
        "typed first-order term matching with a complete substitution result",
    ),
    OperationAdmission(
        "term_rewriting.rewrite_step.compute",
        AdmissionDecision.KEEP,
        "explicit one-step rewrite choices with a typed rewrite application result",
    ),
    OperationAdmission(
        "term_rewriting.unification.compute",
        AdmissionDecision.KEEP,
        "most general unifier certificate for a bounded first-order unification problem",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
