"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.integral_binary_quadratic_forms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.binary_quadratic_form.check",
        AdmissionDecision.KEEP,
        "exact primitive positive-definite form check returning the canonical form value",
    ),
    OperationAdmission(
        "number_theory.binary_quadratic_form.evaluate",
        AdmissionDecision.KEEP,
        "exact evaluation of a binary quadratic form at an integer pair",
    ),
    OperationAdmission(
        "number_theory.binary_quadratic_form.reduce",
        AdmissionDecision.KEEP,
        "exact Gauss reduction with one replayable SL_2(Z) witness",
    ),
    OperationAdmission(
        "number_theory.binary_quadratic_form.proper_equivalence.decide",
        AdmissionDecision.KEEP,
        "exact proper equivalence decision via canonical reduced representative",
    ),
    OperationAdmission(
        "number_theory.binary_quadratic_form.representations.compute",
        AdmissionDecision.KEEP,
        "complete ordered signed representation set with a discriminant-derived finite search bound",
    ),
    OperationAdmission(
        "number_theory.binary_quadratic_form.reduced_classes.compute",
        AdmissionDecision.KEEP,
        "exact complete enumeration of reduced classes for a negative discriminant",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
