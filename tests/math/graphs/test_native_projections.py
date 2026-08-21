from __future__ import annotations

import networkx as nx

from jacobian.math.graphs import (
    biconnected_components,
    radius,
    strongly_connected_components,
)


def test_graph_projections_remain_available_without_catalog_slots() -> None:
    undirected = nx.path_graph(4)
    directed = nx.DiGraph([(0, 1), (1, 0), (1, 2)])

    assert radius(undirected) == 2
    assert biconnected_components(undirected) == (
        frozenset({2, 3}),
        frozenset({1, 2}),
        frozenset({0, 1}),
    )
    assert strongly_connected_components(directed) == (
        frozenset({2}),
        frozenset({0, 1}),
    )
