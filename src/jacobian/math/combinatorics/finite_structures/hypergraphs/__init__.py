"""Supported native finite-hypergraph API."""

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    EdgeIntersectionEntry,
    EdgeIntersectionsResult,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    clique_expansion,
    dual,
    edge_intersection_graph,
    edge_intersections,
    incidence_graph,
    independence_number,
    induced_type_profile,
    maximum_edge_matching,
    maximum_weight_packing,
    minimum_transversal,
    parameters,
    verify_independence_number,
    verify_maximum_edge_matching,
    verify_minimum_transversal,
    verify_weighted_packing,
    vertex_degrees,
)

__all__ = [
    "EdgeIntersectionEntry",
    "EdgeIntersectionsResult",
    "FiniteHypergraph",
    "clique_expansion",
    "dual",
    "edge_intersection_graph",
    "edge_intersections",
    "incidence_graph",
    "independence_number",
    "induced_type_profile",
    "maximum_edge_matching",
    "maximum_weight_packing",
    "minimum_transversal",
    "parameters",
    "verify_independence_number",
    "verify_maximum_edge_matching",
    "verify_minimum_transversal",
    "verify_weighted_packing",
    "vertex_degrees",
]
