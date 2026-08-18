"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.map.compose",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.map.evaluate",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.map.jacobian",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)
