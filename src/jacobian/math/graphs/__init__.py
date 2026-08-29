"""Supported exact finite simple-graph API."""

from jacobian.math.graphs.independence import (
    IndependenceNumberResult,
    independence_number,
)
from jacobian.math.graphs.operations import (
    biconnected_components,
    compose_graphs,
    diameter,
    explicit_graph,
    is_eulerian,
    radius,
    strongly_connected_components,
    triangle_count,
)
from jacobian.math.graphs.values import (
    ColoredUndirectedGraph,
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)

__all__ = [
    "ColoredUndirectedGraph",
    "IndependenceNumberResult",
    "IndexedSimpleUndirectedGraph",
    "SimpleUndirectedGraph",
    "biconnected_components",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "independence_number",
    "is_eulerian",
    "radius",
    "strongly_connected_components",
    "triangle_count",
]
