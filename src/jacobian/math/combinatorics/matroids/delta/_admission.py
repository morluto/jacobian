"""Owner-local admission decisions for finite delta-matroid operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.matroids.delta._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "delta_matroid.from_feasible_sets.compute",
        AdmissionDecision.KEEP,
        "exact exhaustive symmetric-exchange recognition with a canonical finite delta-matroid value or deterministic first obstruction",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
