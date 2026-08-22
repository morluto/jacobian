"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.cohomology_operations._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "cohomology.steenrod_square.compute",
        AdmissionDecision.KEEP,
        "exact Steenrod square computation for simplicial cocycles over GF(2)",
    ),
    OperationAdmission(
        "cohomology.bockstein.compute",
        AdmissionDecision.KEEP,
        "exact Bockstein homomorphism for simplicial cocycles over Z/p",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
