"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.topology.cohomology.operations._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "cohomology.steenrod_square.compute",
        AdmissionDecision.KEEP,
        "exact Steenrod square computation for simplicial cocycles over GF(2)",
    ),
    OperationAdmission(
        "cohomology.bockstein.compute",
        AdmissionDecision.NATIVE_ONLY,
        "every admitted request reduces to the zero cocycle, so execution "
        "returns only the predetermined empty degree-(n+1) cochain; no "
        "reusable mathematical computation beyond recognizing the "
        "already-required zero input. Retain the deterministic helper under "
        "the supported native API until the ambient-complex Bockstein kernel "
        "exists",
        native_symbol="jacobian.math.topology.cohomology.operations.compute_bockstein",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
