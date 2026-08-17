"""Exact graph transform kernels backed by NetworkX."""

from __future__ import annotations

from typing import Any

__all__ = [
    "cartesian_product",
    "complement",
    "graph_power",
    "induced_subgraph",
    "line_graph",
]


def _to_networkx(vertex_count: int, edges: list[tuple[int, int]]) -> Any:
    import networkx as nx

    g: Any = nx.Graph()
    g.add_nodes_from(range(vertex_count))
    g.add_edges_from(edges)
    return g


def _from_networkx(g: Any) -> tuple[int, list[tuple[int, int]]]:
    return (g.number_of_nodes(), list(g.edges()))


def complement(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    g = _to_networkx(vertex_count, edges)
    result = nx_complement(g)
    return _from_networkx(result)


def nx_complement(g: Any) -> Any:
    import networkx as nx

    return nx.complement(g)


def induced_subgraph(
    vertex_count: int, edges: list[tuple[int, int]], vertices: list[int]
) -> tuple[int, list[tuple[int, int]]]:
    import networkx as nx

    g = _to_networkx(vertex_count, edges)
    sub = nx.induced_subgraph(g, vertices)
    old_to_new = {old: new for new, old in enumerate(vertices)}
    result_edges = [(old_to_new[e[0]], old_to_new[e[1]]) for e in sub.edges()]
    return (len(vertices), result_edges)


def line_graph(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    import networkx as nx

    g = _to_networkx(vertex_count, edges)
    lg = nx.line_graph(g)
    lg_nodes = list(lg.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(lg_nodes)}
    result_edges = [(node_to_idx[e[0]], node_to_idx[e[1]]) for e in lg.edges()]
    return (len(lg_nodes), result_edges)


def graph_power(
    vertex_count: int, edges: list[tuple[int, int]], power: int
) -> tuple[int, list[tuple[int, int]]]:
    import networkx as nx

    g = _to_networkx(vertex_count, edges)
    result = nx.power(g, power)
    return _from_networkx(result)


def cartesian_product(
    left_vc: int,
    left_edges: list[tuple[int, int]],
    right_vc: int,
    right_edges: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    import networkx as nx

    g1 = _to_networkx(left_vc, left_edges)
    g2 = _to_networkx(right_vc, right_edges)
    result = nx.cartesian_product(g1, g2)
    return _from_networkx(result)
