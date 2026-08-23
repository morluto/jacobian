"""Owner-local admission decisions for prime-field matrix operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.prime_field_matrix._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "prime_field.matrix.rank.compute",
        AdmissionDecision.KEEP,
        "exact characteristic-dependent rank over an explicit prime field",
    ),
    OperationAdmission(
        "prime_field.matrix.rref.compute",
        AdmissionDecision.KEEP,
        "exact RREF with pivot columns over an explicit prime field",
    ),
    OperationAdmission(
        "prime_field.matrix.nullspace.compute",
        AdmissionDecision.KEEP,
        "exact nullspace basis over an explicit prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
