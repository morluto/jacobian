"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.logic.languages.words._tools import TOOLS

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
    OperationAdmission(
        "substitution.dependency_graph.compute",
        AdmissionDecision.KEEP,
        "exact occurrence-labelled dependency graph of a substitution",
    ),
    OperationAdmission(
        "substitution.primitivity_profile.compute",
        AdmissionDecision.KEEP,
        "complete finite primitivity decision with a least exponent or obstruction",
    ),
    OperationAdmission(
        "substitution.fixed_point_prefix.compute",
        AdmissionDecision.KEEP,
        "exact finite prefix from a certified prolongable substitution",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
