"""NetworkX atlas regression for exact matching certificates."""

from __future__ import annotations

import networkx as nx

from jacobian.domains.graph_optimization import build_graph_invariant_bundle
from jacobian.operations import ComputedSuccess


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
        if operation.capability_id == "graph.invariant.maximum_matching.compute"
    )
    for indexed_graph in nx.graph_atlas_g():
        graph = nx.relabel_nodes(
            indexed_graph,
            {vertex: str(vertex) for vertex in indexed_graph},
        )
        request = operation.request_model.model_validate(
            {"graph": _graph_payload(graph)}
        )
        outcome = operation.implementation(request)
        assert isinstance(outcome, ComputedSuccess)
        result = outcome.value
        barrier = set(result.certificate.barrier_vertices)
        reduced = graph.subgraph(set(graph) - barrier)
        odd_component_count = sum(
            len(component) % 2 for component in nx.connected_components(reduced)
        )

        assert result.certificate.odd_component_count == odd_component_count
        assert 2 * result.maximum_matching_cardinality == (
            len(graph) + len(barrier) - odd_component_count
        )
