"""Exact graph transform kernels backed by NetworkX."""

from __future__ import annotations

from typing import Any

__all__ = ["complement", "graph_power", "induced_subgraph", "line_graph"]


def _to_networkx(vertex_count: int, edges: list[tuple[int, int]]) -> Any:
    import networkx as nx

    graph: Any = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    graph.add_edges_from(edges)
    return graph


def _from_networkx(graph: Any) -> tuple[int, list[tuple[int, int]]]:
    return (graph.number_of_nodes(), list(graph.edges()))


def complement(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the complement of a simple graph on vertices 0..vertex_count-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    return _from_networkx(nx.complement(graph))


def induced_subgraph(
    vertex_count: int, edges: list[tuple[int, int]], vertices: list[int]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the induced subgraph on ``vertices``, reindexed 0..len-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    subgraph = nx.induced_subgraph(graph, vertices)
    old_to_new = {old: new for new, old in enumerate(vertices)}
    result_edges = [
        (old_to_new[source], old_to_new[target]) for source, target in subgraph.edges()
    ]
    return (len(vertices), result_edges)


def line_graph(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the line graph with vertices reindexed 0..|E(G)|-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    transformed = nx.line_graph(graph)
    node_to_index = {node: index for index, node in enumerate(transformed.nodes())}
    result_edges = [
        (node_to_index[source], node_to_index[target])
        for source, target in transformed.edges()
    ]
    return (len(transformed.nodes()), result_edges)


def graph_power(
    vertex_count: int, edges: list[tuple[int, int]], power: int
) -> tuple[int, list[tuple[int, int]]]:
    """Return the ``power``-th power of a simple graph."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    return _from_networkx(nx.power(graph, power))
