"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.hochschild_complexes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "hochschild.chain_complex.compute",
        AdmissionDecision.KEEP,
        "exact Hochschild chain complex from algebra structure constants with trivial bimodule",
    ),
    OperationAdmission(
        "hochschild.homology.compute",
        AdmissionDecision.KEEP,
        "exact Hochschild homology via Gaussian elimination over a prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
