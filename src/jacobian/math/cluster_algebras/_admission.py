"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.cluster_algebras._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "cluster_algebra.seed.mutate.compute",
        AdmissionDecision.KEEP,
        "exact Fomin-Zelevinsky seed mutation for skew-symmetrizable exchange matrices",
    ),
    OperationAdmission(
        "cluster_algebra.g_vector.compute",
        AdmissionDecision.KEEP,
        "exact initial g-vector matrix for cluster seeds with principal coefficients",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
