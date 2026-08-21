"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.prime_field_matrix_ops._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "prime_field_matrix.rank.compute",
        AdmissionDecision.KEEP,
        "exact matrix rank over an explicit prime field via DomainMatrix Gaussian elimination",
    ),
    OperationAdmission(
        "prime_field_matrix.rref.compute",
        AdmissionDecision.KEEP,
        "exact reduced row-echelon form and pivot columns over an explicit prime field",
    ),
    OperationAdmission(
        "prime_field_matrix.nullspace.compute",
        AdmissionDecision.KEEP,
        "exact deterministic nullspace basis over an explicit prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
