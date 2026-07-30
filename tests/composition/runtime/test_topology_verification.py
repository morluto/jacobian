from __future__ import annotations

from typing import Any

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_PRESENTATION = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}


def _computed_results(authorized_complete_runtime) -> tuple[Any, Any, Any]:
    materialized = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.materialize",
            input=_PRESENTATION,
        )
    )
    complex_ = materialized.output["result"]["complex"]
    chain = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input={
                "complex": complex_,
                "coefficient_ring": "PRIME_FIELD",
                "prime": 2,
                "convention": "UNREDUCED",
            },
        )
    )
    homology = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.compute",
            input={
                "complex": complex_,
                "prime": 2,
                "convention": "UNREDUCED",
            },
        )
    )
    return materialized, chain, homology


@pytest.mark.parametrize("result_index", (0, 1, 2))
def test_topology_results_are_independently_verified(
    authorized_complete_runtime,
    result_index: int,
) -> None:
    computed = _computed_results(authorized_complete_runtime)[result_index]

    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert len(verified.artifact_uris) == 4


def test_topology_checker_rejects_forged_cycle_evidence(
    authorized_complete_runtime,
) -> None:
    homology = _computed_results(authorized_complete_runtime)[2]
    result_artifact = authorized_complete_runtime.core.store.get(
        homology.output["result_uri"]
    )
    forged_payload = dict(result_artifact.payload)
    forged_groups = [dict(group) for group in forged_payload["groups"]]
    forged_group = dict(forged_groups[1])
    forged_cycles = [dict(vector) for vector in forged_group["cycle_basis"]]
    forged_cycles[0]["coefficients"] = [1, 0, 0]
    forged_group["cycle_basis"] = forged_cycles
    forged_groups[1] = forged_group
    forged_payload["groups"] = forged_groups
    forged = authorized_complete_runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=forged_payload,
        summary="adversarial noncycle simplicial-homology candidate",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_integral_homology_has_a_dedicated_independent_verifier(
    authorized_complete_runtime,
) -> None:
    materialized = _computed_results(authorized_complete_runtime)[0]
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={
                "complex": materialized.output["result"]["complex"],
                "convention": "UNREDUCED",
            },
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_integral_homology_checker_rejects_a_forged_free_generator(
    authorized_complete_runtime,
) -> None:
    materialized = _computed_results(authorized_complete_runtime)[0]
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={
                "complex": materialized.output["result"]["complex"],
                "convention": "UNREDUCED",
            },
        )
    )
    artifact = authorized_complete_runtime.core.store.get(computed.output["result_uri"])
    payload = dict(artifact.payload)
    groups = [dict(group) for group in payload["groups"]]
    group = dict(groups[1])
    generators = [dict(item) for item in group["free_generators"]]
    cycle = dict(generators[0]["cycle"])
    coefficients = list(cycle["coefficients"])
    coefficients[0] = str(int(coefficients[0]) + 1)
    cycle["coefficients"] = coefficients
    generators[0]["cycle"] = cycle
    group["free_generators"] = generators
    groups[1] = group
    payload["groups"] = groups
    forged = authorized_complete_runtime.core.artifacts.put(
        schema_uri=artifact.manifest.schema_uri,
        semantics_uri=artifact.manifest.semantics_uri,
        parents=artifact.manifest.parents,
        payload=payload,
        summary="adversarial integral-homology free generator",
    )
    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_topology_checker_runtime_binds_only_independent_source(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        item
        for item in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "topology.result.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.topology-exact-checker-source"}
