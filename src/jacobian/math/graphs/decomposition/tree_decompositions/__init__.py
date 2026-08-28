"""Supported native tree-decomposition API."""

from jacobian.math.graphs.decomposition.tree_decompositions.operations import (
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
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
    "vertex_occurrences",
    "width",
]
