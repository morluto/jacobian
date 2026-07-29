from __future__ import annotations

from typing import Any

import pytest
from tests.helpers.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.runtime.model import JacobianRuntime


def _poly(*coefficients_ascending: int) -> dict[str, object]:
    return {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": _q(coefficient), "exponents": [exponent]}
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


def _poly_xy(*terms: tuple[tuple[int, int], int]) -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(coefficient),
                    "exponents": list(exponents),
                }
                for exponents, coefficient in terms
            ]
        },
    }


def _install_verification(
    runtime: JacobianRuntime, *, authorize: bool
) -> tuple[object, ...]:
    adapters, _ = install_exact_domain_verification(
        runtime.core.store,
        runtime.core.schemas,
        runtime.core.artifacts,
        runtime.services.verification,
        runtime.core.checkers,
        polynomial=runtime.portfolio.domain_bundles["polynomial"],
        matrix=runtime.portfolio.domain_bundles["matrix"],
        probability=runtime.portfolio.domain_bundles.get("probability"),
        authorize=authorize,
    )
    for adapter in adapters:
        runtime.core.capabilities.register(adapter)
    return adapters


def _computed_gcd(runtime: JacobianRuntime):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.gcd",
            input={
                "left": _poly(-1, 0, 1),
                "right": _poly(0, 1, 1),
            },
        )
    )


_FAIR_BIT = {
    "atoms": [
        {"value": _q(0), "probability": _q(1, 2)},
        {"value": _q(1), "probability": _q(1, 2)},
    ]
}


@pytest.mark.parametrize(
    ("capability_id", "payload"),
    (
        (
            "probability.finite_distribution.raw_moment.compute",
            {"atoms": _FAIR_BIT["atoms"], "order": 2},
        ),
        (
            "probability.finite_distribution.event_probability.compute",
            {"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        ),
        (
            "probability.finite_distribution.condition.compute",
            {"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        ),
        (
            "probability.finite_distribution.pushforward.compute",
            {
                "distribution": _FAIR_BIT,
                "mapping": [
                    {"source": _q(0), "target": _q(0)},
                    {"source": _q(1), "target": _q(0)},
                ],
            },
        ),
        (
            "probability.finite_distribution.convolution.compute",
            {"left": _FAIR_BIT, "right": _FAIR_BIT},
        ),
    ),
)
def test_probability_results_are_independently_replayed(
    runtime,
    capability_id: str,
    payload: dict[str, Any],
) -> None:
    _install_verification(runtime, authorize=True)
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == capability_id
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_probability_checker_rejects_forged_event_mass(runtime) -> None:
    _install_verification(runtime, authorize=True)
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("probability.finite_distribution.event_probability.compute"),
            input={"distribution": _FAIR_BIT, "event_values": [_q(1)]},
        )
    )
    result_artifact = runtime.core.store.get(computed.output["result_uri"])
    false_payload = dict(result_artifact.payload)
    false_payload["event_probability"] = _q(1)
    false_payload["selected_atoms"] = _FAIR_BIT["atoms"]
    false_result = runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false finite-event probability",
    )

    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="probability.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_public_seam_verifies_exact_producer_result(runtime) -> None:
    adapters = _install_verification(runtime, authorize=True)
    provider_runtime = adapters[0].descriptor.provider_runtime
    assert provider_runtime is not None
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}
    computed = _computed_gcd(runtime)

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "polynomial.compute.gcd"
    assert verified.output["result_uri"] == computed.output["result_uri"]
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert len(verified.artifact_uris) == 4


def test_public_seam_rejects_validly_shaped_false_result(runtime) -> None:
    _install_verification(runtime, authorize=True)
    computed = _computed_gcd(runtime)
    input_uri = computed.output["input_uri"]
    installed = runtime.portfolio.domain_bundles["polynomial"]
    false_result = runtime.core.artifacts.put(
        schema_uri=installed.result_schema_uris["polynomial.compute.gcd"],
        semantics_uri=installed.semantics_uri,
        parents=(input_uri,),
        payload={
            "gcd": _poly(1),
            "bezout": {
                "left_multiplier": _poly(),
                "right_multiplier": _poly(),
            },
            "normalization": "MONIC",
        },
        summary="adversarial false GCD candidate",
    )

    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_public_seam_reports_valid_multivariate_result_as_unsupported(
    runtime,
) -> None:
    _install_verification(runtime, authorize=True)
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.resultant",
            input={
                "left": _poly_xy(((1, 0), 1), ((0, 1), 1)),
                "right": _poly_xy(((1, 0), 1), ((0, 0), 1)),
                "elimination_variable": "x",
            },
        )
    )

    checked = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "UNSUPPORTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["witness_uri"] is None
    assert checked.output["verification_record_uri"] is None
    assert checked.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_induced_tree_result_is_domain_bound_and_independently_replayed(
    runtime_with_references,
) -> None:
    computed = runtime_with_references.core.capabilities.invoke(
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

    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            mode=CapabilityMode.VERIFY,
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

    result_artifact = runtime_with_references.core.store.get(result_uri)
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
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false maximum induced-tree result",
    )
    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_maximum_matching_result_uses_independent_tutte_berge_replay(
    runtime_with_references,
) -> None:
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={
                "graph": {
                    "vertices": ["center", "x", "y", "z"],
                    "edges": [
                        ["center", "x"],
                        ["center", "y"],
                        ["center", "z"],
                    ],
                }
            },
        )
    )
    result_uri = computed.artifact_uris[1]

    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": result_uri},
        )
    )

    assert computed.capability_version == "2"
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
        for descriptor in runtime_with_references.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "graph.invariant.maximum_matching.verify"
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.graph-exact-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.graph-exact-checker-source"}

    result_artifact = runtime_with_references.core.store.get(result_uri)
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={
            "maximum_matching_cardinality": 0,
            "witness_edges": [],
            "certificate": {
                **result_artifact.payload["certificate"],
                "upper_bound": 0,
            },
        },
        summary="adversarial feasible but nonmaximum matching result",
    )
    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


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
    runtime_with_references,
    producer_id: str,
    verifier_id: str,
    result_field: str,
    expected: int,
) -> None:
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                }
            },
        )
    )

    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
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
        for descriptor in runtime_with_references.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.graph-exact-checkers"

    result_artifact = runtime_with_references.core.store.get(
        computed.output["result_uri"]
    )
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={**result_artifact.payload, result_field: 0},
        summary=f"adversarial false {result_field} result",
    )
    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_distance_matrix_result_uses_independent_all_sources_bfs_replay(
    runtime_with_references,
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
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input={"graph": graph},
        )
    )
    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.distance_matrix.compute"
    assert verified.output["result_uri"] == computed.output["result_uri"]
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert len(verified.artifact_uris) == 4

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

    result_artifact = runtime_with_references.core.store.get(
        computed.output["result_uri"]
    )
    order = len(result_artifact.payload["vertices"])
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={
            **result_artifact.payload,
            "distances": [
                [0 if source == target else 1 for target in range(order)]
                for source in range(order)
            ],
        },
        summary="adversarial metric-shaped but false graph distance matrix",
    )
    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is not CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize("value", ("360", "-360", "1", "-1", "101"))
def test_prime_factorization_result_uses_independent_python_flint_replay(
    runtime_with_references,
    value: str,
) -> None:
    producer_id = "integer.compute.prime_factorization"
    verifier_id = "integer.prime_factorization.verify"
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input={"value": value},
        )
    )

    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    provider_runtime = next(
        descriptor.provider_runtime
        for descriptor in runtime_with_references.core.capabilities.catalog().capabilities
        if descriptor.capability_id == verifier_id
    )
    assert provider_runtime is not None
    assert provider_runtime.provider == "jacobian.exact-domain-checkers"
    assert {
        component["provider"]
        for component in provider_runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}


def test_prime_factorization_verifier_rejects_incomplete_factor_list(
    runtime_with_references,
) -> None:
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.compute.prime_factorization",
            input={"value": "360"},
        )
    )
    result_artifact = runtime_with_references.core.store.get(
        computed.output["result_uri"]
    )
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={"factors": result_artifact.payload["factors"][:-1]},
        summary="adversarial incomplete prime factorization result",
    )

    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.prime_factorization.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


@pytest.mark.parametrize("value", ("1", "72", "12", "30"))
def test_powerful_number_result_uses_independent_python_flint_replay(
    runtime_with_references,
    value: str,
) -> None:
    producer_id = "integer.decide.powerful"
    verifier_id = "integer.powerful.verify"
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=producer_id,
            input={"value": value},
        )
    )

    verified = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=verifier_id,
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == producer_id
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_powerful_number_verifier_rejects_schema_valid_wrong_factor_product(
    runtime_with_references,
) -> None:
    computed = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.decide.powerful",
            input={"value": "72"},
        )
    )
    result_artifact = runtime_with_references.core.store.get(
        computed.output["result_uri"]
    )
    false_result = runtime_with_references.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [
                {"prime": "2", "power": 2},
                {"prime": "3", "power": 2},
            ],
            "violating_primes": [],
        },
        summary="adversarial wrong powerful-number factor product",
    )

    rejected = runtime_with_references.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="integer.powerful.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_operator_can_leave_exact_result_verification_unavailable(
    runtime,
) -> None:

    adapters = _install_verification(runtime, authorize=False)

    assert adapters == ()
    assert {"polynomial.result.verify", "matrix.result.verify"}.isdisjoint(
        {
            descriptor.capability_id
            for descriptor in runtime.core.capabilities.catalog().capabilities
        }
    )
