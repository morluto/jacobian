"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.greedoids._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "greedoid.recognize.compute",
        AdmissionDecision.KEEP,
        "exact exhaustive accessibility and exchange recognition with deterministic first obstruction",
    ),
    OperationAdmission(
        "greedoid.rank.compute",
        AdmissionDecision.KEEP,
        "exact greedoid rank over the complete feasible family",
    ),
    OperationAdmission(
        "greedoid.bases.compute",
        AdmissionDecision.KEEP,
        "exact maximal feasible-set family with the common rank",
    ),
    OperationAdmission(
        "greedoid.basic_word.profile.compute",
        AdmissionDecision.KEEP,
        "exact prefix-feasibility profile with first obstruction",
    ),
    OperationAdmission(
        "greedoid.convex_geometry.compute",
        AdmissionDecision.KEEP,
        "exact complementary closed-set family and feasible->closed complement map",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
