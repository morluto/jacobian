"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.chain_complexes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "homological_algebra.chain_complex.homology.compute",
        AdmissionDecision.KEEP,
        "exact chain complex homology via the shared prime-field rank kernel",
    ),
    OperationAdmission(
        "homological_algebra.chain_complex.mapping_cone.compute",
        AdmissionDecision.KEEP,
        "exact mapping cone construction for chain maps over a prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
