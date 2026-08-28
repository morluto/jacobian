"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.sum_of_squares._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.sos.decomposition.check",
        AdmissionDecision.KEEP,
        "exact SOS decomposition checker by coefficient identity over QQ",
    ),
    OperationAdmission(
        "polynomial.sos.gram.check",
        AdmissionDecision.KEEP,
        "exact Gram certificate checker proving symmetry, reconstruction, and PSD by rational arithmetic",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
