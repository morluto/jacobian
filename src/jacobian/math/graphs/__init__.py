"""Supported exact finite simple-graph API."""

from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
    IndependenceNumberResult,
    independence_number,
)
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
    "IndependenceNumberBudget",
    "IndependenceNumberRequest",
    "IndependenceNumberResult",
    "SimpleUndirectedGraph",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "independence_number",
    "is_eulerian",
    "triangle_count",
]
