"""Owner-local admission for the polygon visibility-kernel operation."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry.polygon_kernel._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "geometry.polygon.visibility_kernel.compute",
        AdmissionDecision.KEEP,
        "complete exact source-bound intersection of all oriented polygon-edge half-planes, including lower-dimensional kernels and rational area measures",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
