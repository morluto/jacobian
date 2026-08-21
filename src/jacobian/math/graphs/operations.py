"""Deterministic operations on NetworkX undirected simple graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jacobian.math.graphs.values import GraphCompositionInput, SimpleUndirectedGraph

if TYPE_CHECKING:
    import networkx as nx

__all__ = [
    "biconnected_components",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "is_eulerian",
    "radius",
    "strongly_connected_components",
    "triangle_count",
]


def biconnected_components(graph: nx.Graph[Any]) -> tuple[frozenset[Any], ...]:
    """Return the biconnected vertex components of a simple graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.biconnected_components(graph)


def strongly_connected_components(
    graph: nx.DiGraph[Any],
) -> tuple[frozenset[Any], ...]:
    """Return the strongly connected components of a simple digraph."""

    from jacobian.math.graphs import _networkx

    return _networkx.strongly_connected_components(graph)


def triangle_count(graph: nx.Graph[Any]) -> int:
    """Count the triangles in an undirected simple graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.triangle_count(graph)


def diameter(graph: nx.Graph[Any]) -> int:
    """Return graph diameter, requiring a nonempty connected graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.diameter(graph)


def radius(graph: nx.Graph[Any]) -> int:
    """Return graph radius, requiring a nonempty connected graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.radius(graph)


def is_eulerian(graph: nx.Graph[Any]) -> bool:
    """Return whether the graph has an Eulerian circuit."""

    from jacobian.math.graphs import _networkx

    return _networkx.is_eulerian(graph)


def explicit_graph(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> SimpleUndirectedGraph:
    """Return the immutable canonical graph for explicit graph components."""

    return SimpleUndirectedGraph(
        vertices=tuple(sorted(vertices)),
        edges=tuple(
            sorted(
                (left, right) if left < right else (right, left)
                for left, right in edges
            )
        ),
    )


def compose_graphs(value: GraphCompositionInput) -> SimpleUndirectedGraph:
    """Apply one composition to immutable graph values through NetworkX."""

    from jacobian.math.graphs import _networkx

    return _networkx.compose_graphs(value)
