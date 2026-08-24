"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matrices.canonical_forms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.minimal_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.polynomial.evaluate.compute",
        AdmissionDecision.KEEP,
        "exact polynomial functional calculus is a reusable matrix map and the "
        "source-bound result supplies material reconstruction leverage beyond "
        "scalar polynomial evaluation or caller-authored matrix products",
    ),
    OperationAdmission(
        "matrix.primary_decomposition.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "matrix.rational_canonical_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
