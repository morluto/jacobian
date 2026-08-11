from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.posets import build_finite_poset_bundle

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


@pytest.fixture
def poset_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install finite posets and their independent exact checkers only."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_finite_poset_bundle(),
    ) as services:
        yield services


def _result_payload(runtime: Any, computed: Any) -> dict[str, Any]:
    if "result_uri" in computed.output:
        return runtime.core.store.get(computed.output["result_uri"]).payload
    return computed.output["result"]


def _computed_cases(poset_services) -> list[tuple[str, dict, Any]]:
    materialized = poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.compute",
            input=_PRESENTATION,
        )
    )
    poset = _result_payload(poset_services, materialized)["poset"]
    width_input = {"poset": poset}
    width = poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input=width_input,
        )
    )
    linear_input = {"poset": poset}
    linear = poset_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.linear_extensions.count",
            input=linear_input,
        )
    )
    mobius_input = {"poset": poset, "scope": "COMPLETE_MATRIX", "intervals": []}
    mobius = poset_services.core.capabilities.invoke(
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
    poset_services,
) -> None:
    for producer_id, producer_input, computed in _computed_cases(poset_services):
        verified = poset_services.core.capabilities.invoke(
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
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED, (
            producer_id
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED, producer_id
        assert verified.output["status"] == "VERIFIED", producer_id
        assert verified.output["verification_record_uri"] in verified.artifact_uris, (
            producer_id
        )
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED, (
            producer_id
        )
        assert len(verified.artifact_uris) == (
            4 if producer_id == "poset.linear_extensions.count" else 2
        ), producer_id


def test_poset_checker_rejects_forged_width_certificate(
    poset_services,
) -> None:
    materialized = poset_services.core.capabilities.invoke(
        CapabilityRequest(capability_id="poset.finite.compute", input=_PRESENTATION)
    )
    producer_id = "poset.width.compute"
    producer_input = {"poset": _result_payload(poset_services, materialized)["poset"]}
    width = poset_services.core.capabilities.invoke(
        CapabilityRequest(capability_id=producer_id, input=producer_input)
    )
    forged_candidate = deepcopy(width.output["result"])
    forged_candidate["maximum_antichain"] = ["0", "1"]
    rejected = poset_services.core.capabilities.invoke(
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
    poset_services,
) -> None:
    descriptor = next(
        item
        for item in poset_services.core.capabilities.catalog().capabilities
        if item.capability_id == "poset.width.verify"
    )
    assert descriptor.provider_runtime is not None
    assert {
        component["provider"]
        for component in descriptor.provider_runtime.configuration["components"]
    } == {"jacobian.poset-exact-checker-source"}
