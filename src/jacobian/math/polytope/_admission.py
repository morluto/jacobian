"""Owner-local admission decisions for built-in polytope operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polytope._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polytope.volume.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value with material "
        "computational and reliability leverage: exact rational volume "
        "is a composable building block for Ehrhart positivity and "
        "lattice point counting",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
