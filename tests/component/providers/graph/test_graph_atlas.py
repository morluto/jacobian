from __future__ import annotations

import networkx as nx
import pytest

import jacobian.graphs.atlas as graph_atlas
from jacobian.domains.graph_optimization.invariants import _maximum_matching


def test_graph_atlas_is_built_once_and_cached_as_frozen_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_atlas._graph_atlas_by_order.cache_clear()
    original = nx.graph_atlas_g
    calls = 0

    def counted_graph_atlas() -> list[nx.Graph[int]]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(nx, "graph_atlas_g", counted_graph_atlas)

    first = graph_atlas.graph_atlas_order(7)
    second = graph_atlas.graph_atlas_order(7)

    assert calls == 1
    assert first is second
    assert len(first) == 1044
    assert all(nx.is_frozen(graph) for graph in first)


def test_gallai_edmonds_barrier_certifies_every_graph_through_order_seven() -> None:
    for indexed_graph in nx.graph_atlas_g():
        graph = nx.relabel_nodes(
            indexed_graph,
            {vertex: str(vertex) for vertex in indexed_graph},
        )
        result = _maximum_matching(graph)
        barrier = set(result.certificate.barrier_vertices)
        reduced = graph.subgraph(set(graph) - barrier)
        odd_component_count = sum(
            len(component) % 2 for component in nx.connected_components(reduced)
        )

        assert result.certificate.odd_component_count == odd_component_count
        assert 2 * result.maximum_matching_cardinality == (
            len(graph) + len(barrier) - odd_component_count
        )
