"""Supported exact finite simple-graph API."""

from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
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
from jacobian.math.graphs.transforms import (
    complement,
    graph_power,
    induced_subgraph,
    line_graph,
)
from jacobian.math.graphs.values import (
    ColoredUndirectedGraph,
    GraphCompositionInput,
    SimpleUndirectedGraph,
)

__all__ = [
    "ColoredUndirectedGraph",
    "GraphCompositionInput",
    "IndependenceNumberBudget",
    "IndependenceNumberRequest",
    "IndependenceNumberResult",
    "SimpleUndirectedGraph",
    "biconnected_components",
    "complement",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "graph_power",
    "independence_number",
    "induced_subgraph",
    "is_eulerian",
    "line_graph",
    "radius",
    "strongly_connected_components",
    "triangle_count",
]
