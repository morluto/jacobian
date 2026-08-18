"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

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
