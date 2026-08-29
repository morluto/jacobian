"""Supported native finite-hypergraph API."""

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    EdgeIntersectionEntry,
    EdgeIntersectionsResult,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    clique_expansion,
    dual,
    edge_intersections,
    incidence_graph,
    independence_number,
    induced_type_profile,
    maximum_edge_matching,
    minimum_transversal,
    parameters,
    vertex_degrees,
)

__all__ = [
    "EdgeIntersectionEntry",
    "EdgeIntersectionsResult",
    "FiniteHypergraph",
    "clique_expansion",
    "dual",
    "edge_intersections",
    "incidence_graph",
    "independence_number",
    "induced_type_profile",
    "maximum_edge_matching",
    "minimum_transversal",
    "parameters",
    "vertex_degrees",
]
