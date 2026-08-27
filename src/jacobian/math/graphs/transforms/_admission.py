"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.transforms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.complement.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.complement",
    ),
    OperationAdmission(
        "graph.induced_subgraph.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.induced_subgraph",
    ),
    OperationAdmission(
        "graph.line_graph.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.line_graph",
    ),
    OperationAdmission(
        "graph.power.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.graphs.graph_power",
    ),
    OperationAdmission(
        "graph.path_profile.compute",
        AdmissionDecision.KEEP,
        "distinct complete fixed-length simple path count profile by endpoint pair that a loop of scalar path-finding calls cannot establish as one source-bound value",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
