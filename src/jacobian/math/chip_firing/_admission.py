"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.chip_firing._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.chip_firing.laplacian.compute",
        AdmissionDecision.KEEP,
        "exact graph Laplacian with degree vector and labelled axes",
    ),
    OperationAdmission(
        "graph.chip_firing.fire_vertex.compute",
        AdmissionDecision.KEEP,
        "exact chip-firing action with vertex degree transfer",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
