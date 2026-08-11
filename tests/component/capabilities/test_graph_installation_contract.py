from __future__ import annotations

from tests.component.capabilities.graph_capabilities_support import open_graph_services

from jacobian.graphs import GraphInstallation

_GRAPH_CAPABILITY_VERSIONS = {
    "graph.construct.explicit": "1",
    "graph.search.atlas": "1",
    "graph.compute.properties": "2",
    "graph.realize.degree_sequence": "1",
    "graph.compute.neighborhood_independence": "1",
}


def test_graph_installation_preserves_public_identity_without_checkers(
    tmp_path,
) -> None:
    with open_graph_services(tmp_path / "state") as services:
        assert isinstance(services.graph, GraphInstallation)
        assert services.graph.degree_sequence_checker_id is None
        assert services.graph.neighborhood_checker_id is None

        descriptors = services.core.capabilities.catalog().capabilities
        graph_descriptors = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.capability_id in _GRAPH_CAPABILITY_VERSIONS
        )
        assert tuple(item.capability_id for item in graph_descriptors) == tuple(
            sorted(_GRAPH_CAPABILITY_VERSIONS)
        )
        assert {
            item.capability_id: item.version for item in graph_descriptors
        } == _GRAPH_CAPABILITY_VERSIONS
        for capability_id in (
            "graph.realize.degree_sequence",
            "graph.compute.neighborhood_independence",
        ):
            descriptor = next(
                item for item in descriptors if item.capability_id == capability_id
            )
            assert descriptor.provider_runtime.checker_ids == ()
