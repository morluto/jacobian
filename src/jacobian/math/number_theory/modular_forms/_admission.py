"""Owner-local admission for reviewed level-one modular q-expansions."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "modular_form.level_one.named_q_expansion.compute",
        AdmissionDecision.KEEP,
        "exact normalized q-prefix construction for a closed level-one modular-form family with distinct discovery and composition value",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
