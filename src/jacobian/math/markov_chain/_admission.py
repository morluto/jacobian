"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.markov_chain._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "probability.markov_chain.mixing_time.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded invariant with an explicit incomplete search outcome",
    ),
    OperationAdmission(
        "probability.markov_chain.ergodic.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "probability.markov_chain.stationary_distribution.compute",
        AdmissionDecision.KEEP,
        "returns every canonical extreme point of the complete stationary-distribution simplex",
    ),
    OperationAdmission(
        "probability.markov_chain.communicating_classes.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
