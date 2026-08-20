"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.graphs.tree_decompositions._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "graph.tree_decomposition.width.compute",
        AdmissionDecision.KEEP,
        "exact per-bag cardinality and width of a well-formed tree decomposition",
    ),
    OperationAdmission(
        "graph.tree_decomposition.vertex_occurrences.compute",
        AdmissionDecision.KEEP,
        "exact per-source-vertex occurrence subtrees with induced tree edges",
    ),
    OperationAdmission(
        "graph.tree_decomposition.adhesions.compute",
        AdmissionDecision.KEEP,
        "exact per-tree-edge adhesion profile with maximum adhesion and size profile",
    ),
    OperationAdmission(
        "graph.tree_decomposition.reroot.compute",
        AdmissionDecision.KEEP,
        "exact reroot with parent, children, depth, and root-to-node paths",
    ),
    OperationAdmission(
        "graph.tree_decomposition.restrict.compute",
        AdmissionDecision.KEEP,
        "exact restriction to a source-vertex subset with deterministic cleanup",
    ),
    OperationAdmission(
        "graph.tree_decomposition.bag_intersection_graph.compute",
        AdmissionDecision.KEEP,
        "exact weighted bag-intersection graph projection",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
