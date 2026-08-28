"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.groups.cohomology._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "group_cohomology.cohomology.compute",
        AdmissionDecision.KEEP,
        "exact group cohomology with trivial coefficients via the unnormalized inhomogeneous bar complex over GF(p)",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
