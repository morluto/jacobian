from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.core_capability_harnesses import open_graph_core_services
from tests.support.services import DomainTestServices

from jacobian.artifacts import ArtifactValidationError
from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs import GraphInstallation


@pytest.fixture
def graph_core(
    tmp_path: Path,
) -> Iterator[tuple[DomainTestServices, GraphInstallation]]:
    with open_graph_core_services(tmp_path / "state") as services_and_installation:
        yield services_and_installation


def test_generic_graph_artifacts_use_the_authoritative_bounded_model(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, installation = graph_core
    vertices = [f"v{index:03d}" for index in range(256)]

    accepted = services.core.artifacts.put(
        schema_uri=installation.graph_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={
            "graph_schema_version": "1",
            "vertices": vertices,
            "edges": [],
        },
    )

    assert (
        services.core.store.get(accepted.artifact_uri).payload["vertices"] == vertices
    )
    with pytest.raises(ArtifactValidationError, match="does not match its schema"):
        services.core.artifacts.put(
            schema_uri=installation.graph_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload={
                "graph_schema_version": "1",
                "vertices": [*vertices, "v256"],
                "edges": [],
            },
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b"],
            "edges": [["b", "a"]],
        },
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b"],
            "edges": [["a", "a"]],
        },
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b"],
            "edges": [["a", "c"]],
        },
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b"],
            "edges": [["a", "b"], ["a", "b"]],
        },
    ],
)
def test_generic_graph_artifacts_reject_noncanonical_simple_graphs(
    graph_core: tuple[DomainTestServices, GraphInstallation],
    payload: dict[str, object],
) -> None:
    services, installation = graph_core

    with pytest.raises(ArtifactValidationError, match="does not match its schema"):
        services.core.artifacts.put(
            schema_uri=installation.graph_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=payload,
        )


def test_graph_consumers_reject_forged_malformed_payloads(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, installation = graph_core
    forged = services.core.store.put(
        schema_uri=installation.graph_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={
            "graph_schema_version": "1",
            "vertices": ["a", "b"],
            "edges": [["a", "a"]],
        },
        summary="forged malformed graph",
    )

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={"graph_uri": forged.artifact_uri, "properties": ["order"]},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INCOMPATIBLE_GRAPH_ARTIFACT"


def test_explicit_graph_construction_canonicalizes_and_feeds_graph_capabilities(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, installation = graph_core

    constructed = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.explicit",
            input={
                "vertices": ["c", "a", "b"],
                "edges": [["b", "a"], ["c", "b"]],
            },
        )
    )

    assert constructed.execution.status is ExecutionStatus.COMPLETED
    assert constructed.output["graph"] == {
        "graph_schema_version": "1",
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"]],
    }
    graph_uri = constructed.output["graph_uri"]
    stored = services.core.store.get(graph_uri)
    assert stored.payload == constructed.output["graph"]
    assert stored.manifest.schema_uri == installation.graph_schema_uri
    assert stored.manifest.semantics_uri == installation.semantics_uri
    assert stored.manifest.object_digest == constructed.output["graph_object_digest"]

    properties = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={"graph_uri": graph_uri, "properties": ["order", "tree"]},
        )
    )
    assert properties.execution.status is ExecutionStatus.COMPLETED
    assert properties.output["properties"]["order"]["value"] == 3
    assert properties.output["properties"]["tree"]["value"] is True


@pytest.mark.parametrize(
    "input_payload",
    [
        {"vertices": ["a", "a"], "edges": []},
        {"vertices": ["a"], "edges": [["a", "a"]]},
        {"vertices": ["a"], "edges": [["a", "b"]]},
        {"vertices": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]},
    ],
)
def test_explicit_graph_construction_fails_before_artifact_writes(
    graph_core: tuple[DomainTestServices, GraphInstallation],
    input_payload: dict[str, object],
) -> None:
    services, _installation = graph_core

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.explicit",
            input=input_payload,
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_EXPLICIT_GRAPH"
    assert result.artifact_uris == ()
    assert result.diagnostics[0].details["validation_errors"]


def test_graph_atlas_search_is_bounded_complete_and_replayable(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 5,
                "constraints": {
                    "connected": True,
                    "triangle_free": True,
                    "independence_number": 3,
                },
                "limit": 2,
            },
        )
    )

    assert result.output["match_count"] >= result.output["returned_count"] == 2
    assert result.output["truncated"] is (
        result.output["match_count"] > result.output["returned_count"]
    )
    assert "conclusion" not in result.output

    for candidate in result.output["candidates"]:
        graph_uri = candidate["graph_uri"]
        graph = services.core.store.get(graph_uri)
        assert candidate["graph"] == graph.payload
        assert graph.payload["graph_schema_version"] == "1"
        assert len(graph.payload["vertices"]) == 5
        assert candidate["properties"]["connected"] is True
        assert candidate["properties"]["triangle_count"] == 0
        assert candidate["properties"]["independence_number"] == 3


def test_graph_atlas_search_reports_no_match_without_a_truth_claim(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 4,
                "constraints": {
                    "tree": True,
                    "minimum_edges": 6,
                },
                "limit": 1,
            },
        )
    )

    assert result.output["match_count"] == 0
    assert result.output["candidates"] == []
    assert "conclusion" not in result.output


def test_graph_capabilities_return_actionable_parameter_and_artifact_errors(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core

    invalid_range = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 5,
                "constraints": {
                    "minimum_edges": 5,
                    "maximum_edges": 4,
                },
            },
        )
    )
    assert invalid_range.execution.status is ExecutionStatus.ERROR
    assert invalid_range.diagnostics[0].code == "INVALID_CONSTRAINT_RANGE"
    assert invalid_range.diagnostics[0].path == "constraints/minimum_edges"

    missing_graph = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={
                "graph_uri": "artifact://sha256/" + "f" * 64,
                "properties": ["order"],
            },
        )
    )
    assert missing_graph.execution.status is ExecutionStatus.ERROR
    assert missing_graph.diagnostics[0].code == "GRAPH_ARTIFACT_NOT_FOUND"


def test_graph_property_batch_materializes_exact_computed_artifact(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core
    searched = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 5,
                "constraints": {"tree": True, "maximum_degree": 2},
                "limit": 1,
            },
        )
    )
    graph_uri = searched.output["candidates"][0]["graph_uri"]

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={
                "graph_uri": graph_uri,
                "properties": [
                    "order",
                    "size",
                    "connected",
                    "bipartite",
                    "degree_sequence",
                    "triangle_count",
                    "independence_number",
                ],
            },
        )
    )

    assert result.output["graph_uri"] == graph_uri
    assert result.output["properties"] == {
        "bipartite": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": True,
        },
        "connected": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": True,
        },
        "degree_sequence": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": [2, 2, 2, 1, 1],
        },
        "independence_number": {
            "backend": "networkx.max_weight_clique(complement)",
            "exactness": "EXACT",
            "value": 3,
        },
        "order": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": 5,
        },
        "size": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": 4,
        },
        "triangle_count": {
            "backend": "networkx",
            "exactness": "EXACT",
            "value": 0,
        },
    }
    property_artifact = services.core.store.get(result.output["property_artifact_uri"])
    assert set(property_artifact.manifest.parents) == {
        graph_uri,
        *(binding["artifact_uri"] for binding in result.output["results"]),
    }
    assert property_artifact.payload["registry_version"] == "1"
    assert property_artifact.payload["requested_invariants"] == [
        "bipartite",
        "connected",
        "degree_sequence",
        "independence_number",
        "order",
        "size",
        "triangle_count",
    ]
    for binding in result.output["results"]:
        invariant_artifact = services.core.store.get(binding["artifact_uri"])
        assert invariant_artifact.manifest.parents == (graph_uri,)
        assert invariant_artifact.payload["result"] == binding["result"]


@pytest.mark.parametrize(
    ("order", "expected_status"),
    [(24, "COMPUTED"), (25, "NOT_COMPUTED")],
)
def test_exact_independence_number_stops_at_the_order_boundary(
    graph_core: tuple[DomainTestServices, GraphInstallation],
    order: int,
    expected_status: str,
) -> None:
    services, installation = graph_core
    graph = services.core.artifacts.put(
        schema_uri=installation.graph_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={
            "graph_schema_version": "1",
            "vertices": [f"v{index:02d}" for index in range(order)],
            "edges": [],
        },
    )

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={
                "graph_uri": graph.artifact_uri,
                "properties": ["independence_number"],
            },
        )
    )

    invariant = result.output["results"][0]["result"]
    assert invariant["status"] == expected_status
    if order == 24:
        assert invariant["value"] == 24
    else:
        assert invariant["value"] is None
        assert "limited to graphs of order 24" in invariant["detail"]


def test_graph_counterexample_invariant_batch_reproduces_path_five(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core
    searched = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 5,
                "constraints": {"tree": True, "maximum_degree": 2},
                "limit": 1,
            },
        )
    )
    graph_uri = searched.output["candidates"][0]["graph_uri"]

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={
                "graph_uri": graph_uri,
                "properties": [
                    "average_eccentricity",
                    "diameter",
                    "eccentricities",
                    "girth",
                    "harmonic_index",
                    "havel_hakimi_trace",
                    "radius",
                    "residue",
                    "triangle_frequencies",
                ],
            },
        )
    )

    properties = result.output["properties"]
    assert properties["average_eccentricity"]["value"] == {"num": "16", "den": "5"}
    assert properties["diameter"]["value"] == 4
    assert sorted(properties["eccentricities"]["value"].values()) == [2, 3, 3, 4, 4]
    assert properties["girth"]["value"] is None
    assert properties["harmonic_index"]["value"] == {"num": "7", "den": "3"}
    assert properties["havel_hakimi_trace"]["value"] == [
        [2, 2, 2, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0],
        [0, 0],
    ]
    assert properties["radius"]["value"] == 2
    assert properties["residue"]["value"] == 2
    assert set(properties["triangle_frequencies"]["value"].values()) == {0}


def test_graph_invariant_batch_preserves_unsupported_and_not_applicable_results(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core
    searched = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": 4,
                "constraints": {"connected": False},
                "limit": 1,
            },
        )
    )

    result = services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.properties",
            input={
                "graph_uri": searched.output["candidates"][0]["graph_uri"],
                "properties": [
                    "order",
                    "diameter",
                    "radius",
                    "eccentricities",
                    "average_eccentricity",
                    "made_up_invariant",
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    outcomes = {
        binding["invariant"]: binding["result"] for binding in result.output["results"]
    }
    assert outcomes["order"] == {
        "invariant": "order",
        "status": "COMPUTED",
        "value": 4,
        "exactness": "EXACT",
        "backend": "networkx",
        "detail": None,
    }
    for invariant in (
        "diameter",
        "radius",
        "eccentricities",
        "average_eccentricity",
    ):
        assert outcomes[invariant]["status"] == "NOT_APPLICABLE"
        assert outcomes[invariant]["value"] is None
        assert outcomes[invariant]["exactness"] == "NOT_APPLICABLE"
        assert outcomes[invariant]["backend"] == "networkx"
        assert outcomes[invariant]["detail"]
    assert outcomes["made_up_invariant"] == {
        "invariant": "made_up_invariant",
        "status": "UNSUPPORTED",
        "value": None,
        "exactness": "NOT_APPLICABLE",
        "backend": None,
        "detail": (
            "the invariant is not present in graph.compute.properties "
            "registry version 1"
        ),
    }
    assert set(result.output["properties"]) == {"order"}
    assert result.output["supported_invariants"] == sorted(
        result.output["supported_invariants"]
    )
    assert "made_up_invariant" not in result.output["supported_invariants"]


def test_graph_invariant_registry_is_fixed_and_discoverable(
    graph_core: tuple[DomainTestServices, GraphInstallation],
) -> None:
    services, _installation = graph_core
    descriptor = next(
        item
        for item in services.core.capabilities.catalog().capabilities
        if item.capability_id == "graph.compute.properties"
    )

    assert descriptor.version == "2"
    assert descriptor.input_schema["x-supported-invariants"] == [
        "average_eccentricity",
        "bipartite",
        "connected",
        "degree_sequence",
        "diameter",
        "eccentricities",
        "girth",
        "harmonic_index",
        "havel_hakimi_trace",
        "independence_number",
        "maximum_degree",
        "minimum_degree",
        "order",
        "radius",
        "residue",
        "size",
        "tree",
        "triangle_count",
        "triangle_frequencies",
    ]
