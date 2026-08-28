"""Owner-local admission decisions for finite-frame operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology.frames._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "frame.coherence.compute",
        AdmissionDecision.KEEP,
        "exact normalized pairwise coherence and its maximizing pair",
    ),
    OperationAdmission(
        "frame.gram.compute",
        AdmissionDecision.KEEP,
        "exact Gram matrix of a finite vector family",
    ),
    OperationAdmission(
        "frame.potential.compute",
        AdmissionDecision.KEEP,
        "exact frame potential as a reusable quadratic invariant",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
