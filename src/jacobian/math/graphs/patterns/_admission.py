"""Owner-local admission for induced graph-pattern counts."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.patterns._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.induced_vertex_subset_pattern.count",
        AdmissionDecision.KEEP,
        "distinct exact source-bound count of induced vertex subsets; it is not recoverable from one ordinary embedding without complete automorphism quotienting and search bookkeeping",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
