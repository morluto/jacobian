"""Owner-local publication decisions for subsystem-aware matrix operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matrices.subsystems._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.subsystem.kronecker_product.compute",
        AdmissionDecision.KEEP,
        "returns an axis-bound product value whose ordered subsystem factors are "
        "not present in the generic Kronecker-product result",
    ),
    OperationAdmission(
        "matrix.subsystem.partial_trace.compute",
        AdmissionDecision.KEEP,
        "returns a source-bound exact partial trace over named factors, retaining "
        "the surviving factor order rather than only compatible dimensions",
    ),
    OperationAdmission(
        "matrix.subsystem.psd_order.decide",
        AdmissionDecision.KEEP,
        "returns one exact Loewner-order decision bound to both factorized sources, "
        "with inertia and a replayable negative direction when the order fails",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
