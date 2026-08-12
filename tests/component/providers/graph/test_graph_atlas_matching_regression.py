"""NetworkX atlas regression for exact matching certificates."""

from __future__ import annotations

import networkx as nx

from jacobian.domains.graph_optimization import build_graph_invariant_bundle


def _graph_payload(graph: nx.Graph[str]) -> dict[str, object]:
    return {
        "graph_schema_version": "1",
        "vertices": sorted(graph),
        "edges": sorted(sorted(edge) for edge in graph.edges()),
    }


def test_gallai_edmonds_barrier_certifies_every_graph_through_order_seven() -> None:
    operation = next(
        operation
        for operation in build_graph_invariant_bundle().capabilities
        if operation.spec.operation_id == "graph.invariant.maximum_matching.compute"
    )
    for indexed_graph in nx.graph_atlas_g():
        graph = nx.relabel_nodes(
            indexed_graph,
            {vertex: str(vertex) for vertex in indexed_graph},
        )
        request = operation.spec.request_type.model_validate(
            {"graph": _graph_payload(graph)}
        )
        result = operation.spec.execute(request)
        assert isinstance(result, operation.spec.result_type)
        barrier = set(result.certificate.barrier_vertices)
        reduced = graph.subgraph(set(graph) - barrier)
        odd_component_count = sum(
            len(component) % 2 for component in nx.connected_components(reduced)
        )

        assert result.certificate.odd_component_count == odd_component_count
        assert 2 * result.maximum_matching_cardinality == (
            len(graph) + len(barrier) - odd_component_count
        )
