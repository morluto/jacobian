"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.hypergraphs._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "hypergraph.parameters.compute",
        AdmissionDecision.KEEP,
        "exact vertex count, edge count, rank, corank, uniform size, and "
        "total incidences of a finite hypergraph",
    ),
    OperationAdmission(
        "hypergraph.vertex_degrees.compute",
        AdmissionDecision.KEEP,
        "exact vertex-degree map and degree histogram of a finite hypergraph",
    ),
    OperationAdmission(
        "hypergraph.dual.compute",
        AdmissionDecision.KEEP,
        "exact dual hypergraph transposing vertices and edges",
    ),
    OperationAdmission(
        "hypergraph.incidence_graph.compute",
        AdmissionDecision.KEEP,
        "exact bipartite incidence graph (Levi graph) of a finite hypergraph",
    ),
    OperationAdmission(
        "hypergraph.clique_expansion.compute",
        AdmissionDecision.KEEP,
        "exact 2-section graph where vertices are adjacent if they share a hyperedge",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
