"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.words._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "word.factors.length.compute",
        AdmissionDecision.KEEP,
        "complete bounded factor table of a finite word",
    ),
    OperationAdmission(
        "word.periods.compute",
        AdmissionDecision.KEEP,
        "complete period set with border certificate for a finite word",
    ),
    OperationAdmission(
        "word_morphism.incidence_matrix.compute",
        AdmissionDecision.KEEP,
        "exact incidence matrix of a bounded word morphism",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
