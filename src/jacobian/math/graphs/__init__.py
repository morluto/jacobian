"""Supported exact finite simple-graph API."""

from jacobian.math.graphs.operations import (
    compose_graphs,
    diameter,
    explicit_graph,
    is_eulerian,
    triangle_count,
)
from jacobian.math.graphs.values import GraphCompositionInput, SimpleUndirectedGraph

__all__ = [
    "GraphCompositionInput",
    "SimpleUndirectedGraph",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "is_eulerian",
    "triangle_count",
]
