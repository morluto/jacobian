"""Supported native tree-decomposition API."""

from jacobian.math.graphs.decomposition.tree_decompositions.operations import (
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
    verify_adhesions,
    verify_bag_intersection_graph,
    verify_reroot,
    verify_vertex_occurrences,
    verify_width,
    vertex_occurrences,
    width,
)
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)

__all__ = [
    "TreeDecomposition",
    "adhesions",
    "bag_intersection_graph",
    "reroot",
    "restrict",
    "verify_adhesions",
    "verify_bag_intersection_graph",
    "verify_reroot",
    "verify_vertex_occurrences",
    "verify_width",
    "vertex_occurrences",
    "width",
]
