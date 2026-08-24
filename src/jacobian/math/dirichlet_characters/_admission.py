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
        AdmissionDecision.NATIVE_ONLY,
        "cheap deterministic projection of the retained exact principal-character table, kept through the supported native API",
        native_symbol=(
            "jacobian.math.dirichlet_characters.principal_dirichlet_character_value"
        ),
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
