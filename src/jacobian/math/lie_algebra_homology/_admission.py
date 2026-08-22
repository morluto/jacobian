"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.lie_algebra_homology._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "lie_algebra.chevalley_eilenberg.complex.compute",
        AdmissionDecision.KEEP,
        "exact Chevalley-Eilenberg chain complex from Lie bracket structure constants",
    ),
    OperationAdmission(
        "lie_algebra.homology.compute",
        AdmissionDecision.KEEP,
        "exact Lie algebra homology via Chevalley-Eilenberg complex and Gaussian elimination",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
