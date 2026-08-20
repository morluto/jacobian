"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.regular_languages._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "regular_language.complement.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap deterministic accepting-state projection of a supplied complete DFA",
        native_symbol="jacobian.math.regular_languages.dfa_complement",
    ),
    OperationAdmission(
        "regular_language.count_words.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "regular_language.run.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
