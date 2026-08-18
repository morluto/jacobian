"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "boolean.erasure_noise.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "boolean.fourier_spectrum.compute",
        AdmissionDecision.DROP,
        "duplicate of boolean.fourier.walsh_transform.compute",
    ),
    OperationAdmission(
        "boolean.multilinear_extension.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "boolean.truth_table.compute",
        AdmissionDecision.DROP,
        "echoes a caller-supplied truth table without a new mathematical outcome",
    ),
)
