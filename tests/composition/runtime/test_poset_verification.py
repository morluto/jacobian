from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from jacobian.checker_operations import derive_verification_capability_id
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


def _result_payload(runtime: Any, computed: Any) -> dict[str, Any]:
    if "result_uri" in computed.output:
        return runtime.core.store.get(computed.output["result_uri"]).payload
    return computed.output["result"]


def _computed_cases(authorized_complete_runtime) -> list[tuple[str, dict, Any]]:
    materialized = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=_PRESENTATION,
        )
    )
    poset = _result_payload(authorized_complete_runtime, materialized)["poset"]
    width_input = {"poset": poset}
    width = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input=width_input,
        )
    )
    linear_input = {"poset": poset}
    linear = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.linear_extensions.count",
            input=linear_input,
        )
    )
    mobius_input = {"poset": poset, "scope": "COMPLETE_MATRIX", "intervals": []}
    mobius = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.mobius_function.compute",
            input=mobius_input,
        )
    )
    return [
        ("poset.finite.materialize", _PRESENTATION, materialized),
        ("poset.width.compute", width_input, width),
        ("poset.linear_extensions.count", linear_input, linear),
        ("poset.mobius_function.compute", mobius_input, mobius),
    ]


@pytest.mark.parametrize("result_index", (0, 1, 2, 3))
def test_poset_results_are_independently_verified(
    authorized_complete_runtime,
    result_index: int,
) -> None:
    producer_id, producer_input, computed = _computed_cases(
        authorized_complete_runtime
    )[result_index]
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(producer_id),
            mode=CapabilityMode.VERIFY,
            input=(
                {"input": producer_input, "candidate": computed.output["result"]}
                if producer_id in {"poset.finite.materialize", "poset.width.compute"}
                else {"result_uri": computed.output["result_uri"]}
            ),
        )
    )
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert len(verified.artifact_uris) == 4


def test_poset_checker_rejects_forged_width_certificate(
    authorized_complete_runtime,
) -> None:
    producer_id, producer_input, width = _computed_cases(authorized_complete_runtime)[1]
    forged_candidate = deepcopy(width.output["result"])
    forged_candidate["maximum_antichain"] = ["0", "1"]
    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(producer_id),
            mode=CapabilityMode.VERIFY,
            input={"input": producer_input, "candidate": forged_candidate},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_poset_checker_runtime_binds_only_independent_source(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        item
        for item in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "poset.width.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.poset-exact-checker-source"}
