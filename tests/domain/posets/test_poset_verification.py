from __future__ import annotations

from copy import deepcopy
from typing import Any

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
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


def _computed_cases(verified_poset_services) -> list[tuple[str, dict, Any]]:
    materialized = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.compute",
            input=_PRESENTATION,
        )
    )
    poset = _result_payload(verified_poset_services, materialized)["poset"]
    width_input = {"poset": poset}
    width = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input=width_input,
        )
    )
    linear_input = {"poset": poset}
    linear = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.linear_extensions.count",
            input=linear_input,
        )
    )
    mobius_input = {"poset": poset, "scope": "COMPLETE_MATRIX", "intervals": []}
    mobius = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.mobius_function.compute",
            input=mobius_input,
        )
    )
    return [
        ("poset.finite.compute", _PRESENTATION, materialized),
        ("poset.width.compute", width_input, width),
        ("poset.linear_extensions.count", linear_input, linear),
        ("poset.mobius_function.compute", mobius_input, mobius),
    ]


def test_poset_results_are_independently_verified(
    verified_poset_services,
) -> None:
    for producer_id, producer_input, computed in _computed_cases(
        verified_poset_services
    ):
        verified = verified_poset_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=derive_verification_capability_id(producer_id),
                input=(
                    {"input": producer_input, "candidate": computed.output["result"]}
                    if producer_id
                    in {
                        "poset.finite.compute",
                        "poset.width.compute",
                        "poset.mobius_function.compute",
                    }
                    else {"result_uri": computed.output["result_uri"]}
                ),
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED, producer_id
        assert verified.output["status"] == "VERIFIED", producer_id
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            producer_id
        )
        assert verified.verification_record_uri is not None, producer_id
        assert len(verified.artifact_uris) == (
            4 if producer_id == "poset.linear_extensions.count" else 2
        ), producer_id


def test_poset_checker_rejects_forged_width_certificate(
    verified_poset_services,
) -> None:
    materialized = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(capability_id="poset.finite.compute", input=_PRESENTATION)
    )
    producer_id = "poset.width.compute"
    producer_input = {
        "poset": _result_payload(verified_poset_services, materialized)["poset"]
    }
    width = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=producer_id, input=producer_input)
    )
    forged_candidate = deepcopy(width.output["result"])
    forged_candidate["maximum_antichain"] = ["0", "1"]
    rejected = verified_poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=derive_verification_capability_id(producer_id),
            input={"input": producer_input, "candidate": forged_candidate},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_poset_checker_runtime_binds_only_independent_source(
    verified_poset_services,
) -> None:
    descriptor = next(
        item
        for item in verified_poset_services.core.capabilities.catalog().capabilities
        if item.capability_id == "poset.width.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.poset-exact-checker-source"}
