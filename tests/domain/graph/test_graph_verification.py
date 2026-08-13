from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.graph_distance_cases import (
    c7_strong_c7_graph,
    hoffman_singleton_graph,
)
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
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
    producer_input = {
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
    }
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.compute",
            input=producer_input,
        )
    )
    candidate = computed.output["result"]
    assert candidate["optimum_value"] == 3

    verified = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            input={"input": producer_input, "candidate": candidate},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.induced_tree.maximum.compute"
    assert verified.output["verification_record_uri"] is not None
    assert verified.execution.detail == (
        "independent finite-subset exhaustive replay accepted "
        "graph.induced_tree.maximum.compute"
    )
    assert "FLINT" not in verified.execution.detail

    false_payload = dict(candidate)
    false_payload.update(
        {
            "optimum_value": 4,
            "incumbent_value": 4,
            "lower_bound": 4,
            "upper_bound": 4,
            "witness_vertices": ["a", "b", "c", "d"],
        }
    )
    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            input={"input": producer_input, "candidate": false_payload},
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
    assert verified.verification_record_uri is not None
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
    assert verified.verification_record_uri is not None


def test_graph_metric_result_uses_independent_all_sources_bfs_replay(
    graph_verification_services: DomainTestServices,
) -> None:
    cases = (
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
    )
    producer_input = {
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
        }
    }
    for producer_id, verifier_id, result_field, expected in cases:
        computed = graph_verification_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=producer_id,
                input=producer_input,
            )
        )

        verified = graph_verification_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=verifier_id,
                input={
                    "input": producer_input,
                    "candidate": computed.output["result"],
                },
            )
        )

        assert computed.output["result"][result_field] == expected, producer_id
        assert verified.execution.status is ExecutionStatus.COMPLETED, producer_id
        assert verified.output["status"] == "VERIFIED", producer_id
        assert verified.output["operation_id"] == producer_id, producer_id
        assert verified.output["verification_record_uri"] is not None, producer_id
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            producer_id
        )
        provider_runtime = next(
            descriptor.provider_runtime
            for descriptor in graph_verification_services.core.capabilities.catalog().capabilities
            if descriptor.capability_id == verifier_id
        )
        assert provider_runtime is not None, producer_id
        assert provider_runtime.provider == "jacobian.graph-exact-checkers", producer_id

        false_candidate = deepcopy(computed.output["result"])
        false_candidate[result_field] = 0
        rejected = graph_verification_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=verifier_id,
                input={"input": producer_input, "candidate": false_candidate},
            )
        )

        assert rejected.execution.status is ExecutionStatus.COMPLETED, producer_id
        assert rejected.output["status"] == "REJECTED", producer_id
        assert rejected.output["conclusion"] == "UNKNOWN", producer_id
        assert rejected.output["verification_record_uri"] is None, producer_id


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

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.distance_matrix.compute"
    assert verified.output["verification_record_uri"] is not None
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


@pytest.mark.parametrize(
    "graph_factory",
    (hoffman_singleton_graph, c7_strong_c7_graph),
    ids=("hoffman-singleton-order-50", "c7-strong-c7-order-49"),
)
def test_distance_matrix_checker_verifies_proof_critical_large_cases(
    graph_verification_services: DomainTestServices,
    graph_factory,
) -> None:
    producer_input = {"graph": graph_factory()}
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

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_distance_matrix_checker_rejects_one_wrong_edge_distance(
    graph_verification_services: DomainTestServices,
) -> None:
    producer_input = {"graph": hoffman_singleton_graph()}
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input=producer_input,
        )
    )
    candidate = deepcopy(computed.output["result"])
    left, right = producer_input["graph"]["edges"][0]
    left_index = candidate["vertices"].index(left)
    right_index = candidate["vertices"].index(right)
    candidate["distances"][left_index][right_index] = 2
    candidate["distances"][right_index][left_index] = 2

    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            input={"input": producer_input, "candidate": candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_distance_matrix_checker_rejects_candidate_rebound_to_changed_graph(
    graph_verification_services: DomainTestServices,
) -> None:
    original_graph = c7_strong_c7_graph()
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input={"graph": original_graph},
        )
    )
    changed_graph = deepcopy(original_graph)
    changed_graph["edges"].pop(0)

    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            input={
                "input": {"graph": changed_graph},
                "candidate": computed.output["result"],
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda candidate: candidate.pop("vertices"),
        lambda candidate: candidate.update(
            vertices=list(reversed(candidate["vertices"]))
        ),
    ),
    ids=("missing-label-binding", "noncanonical-label-binding"),
)
def test_distance_matrix_verifier_fails_closed_on_malformed_label_binding(
    graph_verification_services: DomainTestServices,
    mutate,
) -> None:
    producer_input = {
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        }
    }
    computed = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input=producer_input,
        )
    )
    candidate = deepcopy(computed.output["result"])
    mutate(candidate)

    rejected = graph_verification_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            input={"input": producer_input, "candidate": candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.ERROR
    assert rejected.artifact_uris == ()
