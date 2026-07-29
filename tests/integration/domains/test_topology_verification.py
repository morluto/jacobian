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


def _computed_results(runtime) -> tuple[Any, Any, Any]:
    materialized = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.materialize",
            input=_PRESENTATION,
        )
    )
    complex_ = materialized.output["result"]["complex"]
    chain = runtime.core.capabilities.invoke(
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
    homology = runtime.core.capabilities.invoke(
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
    runtime_with_references,
    result_index: int,
) -> None:
    runtime = runtime_with_references
    computed = _computed_results(runtime)[result_index]

    verified = runtime.core.capabilities.invoke(
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
    runtime_with_references,
) -> None:
    runtime = runtime_with_references
    homology = _computed_results(runtime)[2]
    result_artifact = runtime.core.store.get(homology.output["result_uri"])
    forged_payload = dict(result_artifact.payload)
    forged_groups = [dict(group) for group in forged_payload["groups"]]
    forged_group = dict(forged_groups[1])
    forged_cycles = [dict(vector) for vector in forged_group["cycle_basis"]]
    forged_cycles[0]["coefficients"] = [1, 0, 0]
    forged_group["cycle_basis"] = forged_cycles
    forged_groups[1] = forged_group
    forged_payload["groups"] = forged_groups
    forged = runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=forged_payload,
        summary="adversarial noncycle simplicial-homology candidate",
    )

    rejected = runtime.core.capabilities.invoke(
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


def test_topology_checker_runtime_binds_only_independent_source(
    runtime_with_references,
) -> None:
    runtime = runtime_with_references
    descriptor = next(
        item
        for item in runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "topology.result.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.topology-exact-checker-source"}
