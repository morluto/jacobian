"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorial_matrices._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.sign.profile.compute",
        AdmissionDecision.KEEP,
        "exact sign profile with entry counts and row/column sums",
    ),
    OperationAdmission(
        "matrix.hadamard.gram_profile.compute",
        AdmissionDecision.KEEP,
        "exact Gram profile with orthogonality replayed exactly and no floating tolerance",
    ),
    OperationAdmission(
        "matrix.hadamard.normalize.compute",
        AdmissionDecision.KEEP,
        "exact deterministic normalization with sign switches, idempotent",
    ),
    OperationAdmission(
        "matrix.hadamard.determinant_profile.compute",
        AdmissionDecision.KEEP,
        "exact |det H| = n^(n/2) and Gram determinant after exact orthogonality",
    ),
    OperationAdmission(
        "matrix.hadamard.sylvester.compute",
        AdmissionDecision.KEEP,
        "deterministic bounded Sylvester construction with ledger",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
