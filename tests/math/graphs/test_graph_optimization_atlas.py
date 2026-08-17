"""NetworkX atlas regression for exact matching certificates."""

from __future__ import annotations

import networkx as nx
import pytest

from jacobian.math.graphs.optimization._tools import TOOLS


def _graph_payload(graph: nx.Graph[str]) -> dict[str, object]:
    return {
        "graph_schema_version": "1",
        "vertices": sorted(graph),
        "edges": sorted(sorted(edge) for edge in graph.edges()),
    }


def _assert_gallai_edmonds_certificate(graphs: list[nx.Graph[int]]) -> None:
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "graph.invariant.maximum_matching.compute"
    )
    for indexed_graph in graphs:
        graph = nx.relabel_nodes(
            indexed_graph,
            {vertex: str(vertex) for vertex in indexed_graph},
        )
        request = operation.request_type.model_validate(
            {"graph": _graph_payload(graph)}
        )
        result = operation.run(request)
        assert isinstance(result, operation.result_type)
        barrier = set(result.certificate.barrier_vertices)
        reduced = graph.subgraph(set(graph) - barrier)
        odd_component_count = sum(
            len(component) % 2 for component in nx.connected_components(reduced)
        )

        assert result.certificate.odd_component_count == odd_component_count
        assert 2 * result.maximum_matching_cardinality == (
            len(graph) + len(barrier) - odd_component_count
        )


def test_gallai_edmonds_barrier_certifies_every_graph_through_order_six() -> None:
    _assert_gallai_edmonds_certificate(
        [graph for graph in nx.graph_atlas_g() if len(graph) <= 6]
    )


@pytest.mark.exhaustive
def test_gallai_edmonds_barrier_certifies_every_graph_of_order_seven() -> None:
    _assert_gallai_edmonds_certificate(
        [graph for graph in nx.graph_atlas_g() if len(graph) == 7]
    )
