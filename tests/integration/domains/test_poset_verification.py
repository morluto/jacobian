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
    "elements": ["0", "a", "b", "1"],
    "relation": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "interpretation": "COVER_EDGES",
}


def _computed_results(runtime) -> tuple[Any, Any, Any, Any]:
    materialized = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=_PRESENTATION,
        )
    )
    poset = materialized.output["result"]["poset"]
    width = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input={"poset": poset},
        )
    )
    linear = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.linear_extensions.count",
            input={"poset": poset},
        )
    )
    mobius = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.mobius_function.compute",
            input={"poset": poset, "scope": "COMPLETE_MATRIX", "intervals": []},
        )
    )
    return materialized, width, linear, mobius


@pytest.mark.parametrize("result_index", (0, 1, 2, 3))
def test_poset_results_are_independently_verified(
    runtime_with_references,
    result_index: int,
) -> None:
    runtime = runtime_with_references
    computed = _computed_results(runtime)[result_index]
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.result.verify",
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


def test_poset_checker_rejects_forged_width_certificate(
    runtime_with_references,
) -> None:
    runtime = runtime_with_references
    width = _computed_results(runtime)[1]
    result_artifact = runtime.core.store.get(width.output["result_uri"])
    forged_payload = dict(result_artifact.payload)
    forged_payload["maximum_antichain"] = ["0", "1"]
    forged = runtime.core.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=forged_payload,
        summary="adversarial comparable antichain candidate",
    )
    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged.artifact_uri},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_poset_checker_runtime_binds_only_independent_source(
    runtime_with_references,
) -> None:
    descriptor = next(
        item
        for item in runtime_with_references.core.capabilities.catalog().capabilities
        if item.capability_id == "poset.result.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.poset-exact-checker-source"}
