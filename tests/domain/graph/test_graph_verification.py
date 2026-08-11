from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import (
    build_graph_invariant_bundle,
    build_graph_optimization_bundle,
)


@pytest.fixture
def graph_verification_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install the two graph bundles covered by this verification contract."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_graph_optimization_bundle(),
        build_graph_invariant_bundle(),
    ) as services:
        yield services


def test_induced_tree_result_is_domain_bound_and_independently_replayed(
    graph_verification_services: DomainTestServices,
) -> None:
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["a", "b"],
                        ["b", "c"],
                        ["c", "d"],
                        ["d", "a"],
                    ],
                },
                "resource_budget": {
                    "wall_seconds": 5,
                    "max_solver_calls": 33,
                    "max_order": 16,
                },
            },
        )
    )
    assert computed.output["optimum_value"] == 3
    result_uri = computed.artifact_uris[1]

    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            input={"result_uri": result_uri},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.induced_tree.maximum.compute"
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent finite-subset exhaustive replay accepted "
        "graph.induced_tree.maximum.compute"
    )
    assert "FLINT" not in verified.execution.detail

    result_artifact = graph_verification_services.core.store.get(result_uri)
    false_payload = dict(result_artifact.payload)
    false_payload.update(
        {
            "optimum_value": 4,
            "incumbent_value": 4,
            "lower_bound": 4,
            "upper_bound": 4,
            "witness_vertices": ["a", "b", "c", "d"],
        }
    )
    false_result = graph_verification_services.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false maximum induced-tree result",
    )
    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            input={"result_uri": false_result.artifact_uri},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_maximum_matching_result_uses_independent_tutte_berge_replay(
    graph_verification_services: DomainTestServices,
) -> None:
    producer_input = {
        "graph": {
            "vertices": ["center", "x", "y", "z"],
            "edges": [
                ["center", "x"],
                ["center", "y"],
                ["center", "z"],
            ],
        }
    }
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input=producer_input,
        )
    )

    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            input={"input": producer_input, "candidate": computed.output["result"]},
        )
    )

    assert computed.capability_version == "3"
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == (
        "graph.invariant.maximum_matching.compute"
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent Tutte-Berge barrier replay accepted "
        "graph.invariant.maximum_matching.compute"
    )
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in graph_verification_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "graph.invariant.maximum_matching.verify"
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.graph-exact-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.graph-exact-checker-source"}

    false_candidate = deepcopy(computed.output["result"])
    false_candidate.update(
        {
            "maximum_matching_cardinality": 0,
            "witness_edges": [],
            "certificate": {
                **false_candidate["certificate"],
                "upper_bound": 0,
            },
        }
    )
    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            input={"input": producer_input, "candidate": false_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_maximum_matching_verifier_replays_a_64_vertex_certificate(
    graph_verification_services: DomainTestServices,
) -> None:
    vertices = [f"v{index:02d}" for index in range(64)]
    producer_input = {
        "graph": {
            "vertices": vertices,
            "edges": [
                [vertices[index], vertices[index + 1]] for index in range(0, 64, 2)
            ],
        }
    }
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input=producer_input,
        )
    )

    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            input={"input": producer_input, "candidate": computed.output["result"]},
        )
    )

    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize(
    ("producer_id", "verifier_id", "result_field", "expected"),
    (
        (
            "graph.invariant.diameter.compute",
            "graph.invariant.diameter.verify",
            "diameter",
            3,
        ),
        (
            "graph.invariant.radius.compute",
            "graph.invariant.radius.verify",
            "radius",
            2,
        ),
    ),
)
def test_graph_metric_result_uses_independent_all_sources_bfs_replay(
    graph_verification_services: DomainTestServices,
    producer_id: str,
    verifier_id: str,
    result_field: str,
    expected: int,
) -> None:
    producer_input = {
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
        }
    }
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input=producer_input,
        )
    )

    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            input={"input": producer_input, "candidate": computed.output["result"]},
        )
    )

    assert computed.output["result"][result_field] == expected
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in graph_verification_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.graph-exact-checkers"

    false_candidate = deepcopy(computed.output["result"])
    false_candidate[result_field] = 0
    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            input={"input": producer_input, "candidate": false_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_distance_matrix_result_uses_independent_all_sources_bfs_replay(
    graph_verification_services: DomainTestServices,
) -> None:
    graph: dict[str, Any] = {
        "vertices": ["0", "1", "2", "3", "4", "5"],
        "edges": [
            ["0", "3"],
            ["0", "4"],
            ["1", "4"],
            ["2", "4"],
            ["3", "4"],
            ["3", "5"],
        ],
    }
    producer_input = {"graph": graph}
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input=producer_input,
        )
    )
    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            input={"input": producer_input, "candidate": computed.output["result"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.distance_matrix.compute"
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert len(verified.artifact_uris) == 2

    matrix = computed.output["result"]["distances"]
    degree = dict.fromkeys(graph["vertices"], 0)
    for left, right in graph["edges"]:
        degree[left] += 1
        degree[right] += 1
    maximum_degree = max(degree.values())
    maximum_degree_vertices = sorted(
        vertex for vertex, value in degree.items() if value == maximum_degree
    )
    matrix_vertices = computed.output["result"]["vertices"]
    designated_indices = [
        matrix_vertices.index(vertex) for vertex in maximum_degree_vertices
    ]
    distances_to_set = [
        min(matrix[index][designated] for designated in designated_indices)
        for index in range(len(matrix_vertices))
    ]
    maximum_distance = max(distances_to_set)
    maximizing_vertices = [
        vertex
        for vertex, distance in zip(
            matrix_vertices,
            distances_to_set,
            strict=True,
        )
        if distance == maximum_distance
    ]
    assert maximum_degree_vertices == ["4"]
    assert distances_to_set == [1, 1, 1, 1, 0, 2]
    assert maximum_distance == 2
    assert maximizing_vertices == ["5"]

    false_candidate = deepcopy(computed.output["result"])
    order = len(false_candidate["vertices"])
    false_candidate["distances"] = [
        [0 if source == target else 1 for target in range(order)]
        for source in range(order)
    ]
    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            input={"input": producer_input, "candidate": false_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is not CapabilityAssuranceLevel.VERIFIED
