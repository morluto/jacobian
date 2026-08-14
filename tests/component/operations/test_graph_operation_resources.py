from __future__ import annotations

from tests.component.operations.graph_operations_support import open_graph_services

from jacobian.graphs import GraphOperationResources

_GRAPH_OPERATION_VERSIONS = {
    "graph.construct.explicit": "1",
    "graph.search.atlas": "1",
    "graph.compute.properties": "2",
    "graph.realize.degree_sequence": "1",
    "graph.compute.neighborhood_independence": "1",
}


def test_graph_resources_preserve_public_identity_without_checkers(
    tmp_path,
) -> None:
    with open_graph_services(tmp_path / "state") as services:
        assert isinstance(services.graph, GraphOperationResources)
        assert services.graph.degree_sequence_checker_id is None
        assert services.graph.neighborhood_checker_id is None

        descriptors = services.core.operations.snapshot().operations
        graph_descriptors = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.operation_id in _GRAPH_OPERATION_VERSIONS
        )
        assert tuple(item.operation_id for item in graph_descriptors) == tuple(
            sorted(_GRAPH_OPERATION_VERSIONS)
        )
        assert {
            item.operation_id: item.version for item in graph_descriptors
        } == _GRAPH_OPERATION_VERSIONS
        for operation_id in (
            "graph.realize.degree_sequence",
            "graph.compute.neighborhood_independence",
        ):
            descriptor = next(
                item for item in descriptors if item.operation_id == operation_id
            )
            assert descriptor.provider_runtime.checker_ids == ()
