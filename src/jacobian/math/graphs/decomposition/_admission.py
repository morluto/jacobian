"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.decomposition._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.decomposition.spqr_tree.compute",
        AdmissionDecision.KEEP,
        "bounded exact structural decomposition with source-edge ownership,"
        " virtual-edge gluing, and a distinct replayable postcondition",
    ),
    OperationAdmission(
        "graph.decomposition.biconnected_components.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful projection of the retained block-cut-tree decomposition",
        native_symbol="jacobian.math.graphs.biconnected_components",
    ),
    OperationAdmission(
        "graph.decomposition.block_cut_tree.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.decomposition.bridge_block_tree.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "graph.decomposition.ear.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
