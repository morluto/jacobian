"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.hypergraphs._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "hypergraph.independence_number.compute",
        AdmissionDecision.KEEP,
        "distinct source-bound exact-or-unknown maximum subset invariant that "
        "cannot be recovered from clique or incidence graph transforms",
    ),
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
        "hypergraph.edge_intersections.compute",
        AdmissionDecision.KEEP,
        "complete exact source-bound indexed edge-pair intersection ledger, size "
        "histogram, and canonical linearity violation with material reliability "
        "leverage over caller-authored pair enumeration",
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
        "exact 2-section of a finite hypergraph as a canonical "
        "SimpleUndirectedGraph: two vertices are adjacent if they share a "
        "hyperedge",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
