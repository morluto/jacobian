"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.recurrence_solving._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "sequence.recurrence.prime_field.find",
        AdmissionDecision.KEEP,
        "exact minimal LFSR over an explicit prime field via Berlekamp-Massey with material leverage over bespoke recurrence fitting",
    ),
    OperationAdmission(
        "sequence.recurrence.closed_form.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "sequence.recurrence.find",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
