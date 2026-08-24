"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matrices.analysis._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.farkas_certificate.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "matrix.inertia.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symmetric.rational_spectrum_claim.check",
        AdmissionDecision.KEEP,
        "one source-bound exact predicate that replays every rational eigenspace multiplicity and completeness from a canonical symmetric QQ matrix",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
