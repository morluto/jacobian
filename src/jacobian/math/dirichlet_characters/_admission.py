"""Owner-local admission decisions for principal Dirichlet characters."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.dirichlet_characters._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "dirichlet_character.principal.compute",
        AdmissionDecision.KEEP,
        "canonical finite extension-by-zero character table is a reusable exact value for arithmetic and Fourier composition",
    ),
    OperationAdmission(
        "dirichlet_character.principal.value.compute",
        AdmissionDecision.KEEP,
        "source-bound exact character evaluation preserves the supplied canonical character identity",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
