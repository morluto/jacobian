"""Owner-local admission for coherent-configuration operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.designs.coherent_configurations._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "coherent_configuration.analyze.compute",
        AdmissionDecision.KEEP,
        "exact bounded complete pair-partition analysis with source-bound fibres, transpose map, and intersection tensor",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
