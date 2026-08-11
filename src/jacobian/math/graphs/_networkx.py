"""Private NetworkX backend for public graph operations."""

from __future__ import annotations

from typing import Any, cast

import networkx as nx


def simple_graph(graph: nx.Graph[Any]) -> nx.Graph[Any]:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("graph must be undirected and simple")
    if graph.number_of_nodes() > 32:
        raise ValueError("graph may contain at most 32 vertices")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
    return graph


def triangle_count(graph: nx.Graph[Any]) -> int:
    counts = cast(dict[Any, int], nx.triangles(simple_graph(graph)))
    return sum(counts.values()) // 3


def diameter(graph: nx.Graph[Any]) -> int:
    value = simple_graph(graph)
    if not value or not nx.is_connected(value):
        raise ValueError("diameter requires a nonempty connected graph")
    return int(nx.diameter(value))


def is_eulerian(graph: nx.Graph[Any]) -> bool:
    return bool(nx.is_eulerian(simple_graph(graph)))
