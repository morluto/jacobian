"""Deterministic operations on NetworkX undirected simple graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["diameter", "is_eulerian", "triangle_count"]


def triangle_count(graph: nx.Graph[Any]) -> int:
    """Count the triangles in an undirected simple graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.triangle_count(graph)


def diameter(graph: nx.Graph[Any]) -> int:
    """Return graph diameter, requiring a nonempty connected graph."""

    from jacobian.math.graphs import _networkx

    return _networkx.diameter(graph)


def is_eulerian(graph: nx.Graph[Any]) -> bool:
    """Return whether the graph has an Eulerian circuit."""

    from jacobian.math.graphs import _networkx

    return _networkx.is_eulerian(graph)
