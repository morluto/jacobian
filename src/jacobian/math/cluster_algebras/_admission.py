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
        AdmissionDecision.NATIVE_ONLY,
        "identity projection of the initial seed derived solely from n; "
        "retained through the supported native API, with genuine g-vectors "
        "requiring mutation-sequence data that no candidate supplies",
        native_symbol="jacobian.math.cluster_algebras.compute_g_vectors",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
