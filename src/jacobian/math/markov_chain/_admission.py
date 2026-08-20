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
        "probability.markov_chain.ergodic.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "probability.markov_chain.stationary_distribution.compute",
        AdmissionDecision.KEEP,
        "returns every canonical extreme point of the complete stationary-distribution simplex",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
